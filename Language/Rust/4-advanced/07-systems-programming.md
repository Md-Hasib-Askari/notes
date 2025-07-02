# Systems Programming and OS Development

## Overview
Learn low-level systems programming concepts and operating system development using Rust's safety guarantees without sacrificing performance.

## Kernel Development Basics

### No-Std Environment
```rust
#![no_std]
#![no_main]

use core::panic::PanicInfo;

#[no_mangle]
pub extern "C" fn _start() -> ! {
    // Kernel entry point
    loop {}
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

// Custom allocator for kernel space
use linked_list_allocator::LockedHeap;

#[global_allocator]
static ALLOCATOR: LockedHeap = LockedHeap::empty();
```

### Memory Management
```rust
// Page frame allocator
pub struct FrameAllocator {
    next: usize,
    max_frames: usize,
}

impl FrameAllocator {
    pub fn new(start_addr: usize, end_addr: usize) -> Self {
        Self {
            next: start_addr / 4096,
            max_frames: end_addr / 4096,
        }
    }
    
    pub fn allocate_frame(&mut self) -> Option<PhysFrame> {
        if self.next < self.max_frames {
            let frame = PhysFrame::containing_address(PhysAddr::new(self.next * 4096));
            self.next += 1;
            Some(frame)
        } else {
            None
        }
    }
}

// Virtual memory mapping
pub unsafe fn create_example_mapping(
    page: Page,
    mapper: &mut impl Mapper<Size4KiB>,
    frame_allocator: &mut impl FrameAllocator,
) {
    use x86_64::structures::paging::PageTableFlags as Flags;
    
    let frame = PhysFrame::containing_address(PhysAddr::new(0xb8000));
    let flags = Flags::PRESENT | Flags::WRITABLE;
    
    let map_to_result = mapper.map_to(page, frame, flags, frame_allocator);
    map_to_result.expect("map_to failed").flush();
}
```

## Device Drivers

### Hardware Abstraction
```rust
// Serial port driver
pub struct SerialPort {
    data_port: Port<u8>,
    interrupt_enable: Port<u8>,
    line_control: Port<u8>,
    modem_control: Port<u8>,
    line_status: Port<u8>,
}

impl SerialPort {
    pub fn new(base: u16) -> SerialPort {
        SerialPort {
            data_port: Port::new(base),
            interrupt_enable: Port::new(base + 1),
            line_control: Port::new(base + 3),
            modem_control: Port::new(base + 4),
            line_status: Port::new(base + 5),
        }
    }
    
    pub fn init(&mut self) {
        unsafe {
            self.interrupt_enable.write(0x00);
            self.line_control.write(0x80);
            self.data_port.write(0x03);
            self.interrupt_enable.write(0x00);
            self.line_control.write(0x03);
            self.modem_control.write(0x0B);
        }
    }
    
    pub fn send_byte(&mut self, byte: u8) {
        unsafe {
            while (self.line_status.read() & 0x20) == 0 {}
            self.data_port.write(byte);
        }
    }
}

// GPIO abstraction
trait GpioPin {
    fn set_high(&mut self);
    fn set_low(&mut self);
    fn is_high(&self) -> bool;
    fn set_direction(&mut self, direction: Direction);
}

enum Direction {
    Input,
    Output,
}

struct Pin<const N: usize>;

impl<const N: usize> GpioPin for Pin<N> {
    fn set_high(&mut self) {
        // Platform-specific implementation
        unsafe {
            let gpio_base = 0x3F200000 as *mut u32;
            let set_reg = gpio_base.add(7 + N / 32);
            set_reg.write_volatile(1 << (N % 32));
        }
    }
    
    fn set_low(&mut self) {
        unsafe {
            let gpio_base = 0x3F200000 as *mut u32;
            let clear_reg = gpio_base.add(10 + N / 32);
            clear_reg.write_volatile(1 << (N % 32));
        }
    }
    
    fn is_high(&self) -> bool {
        unsafe {
            let gpio_base = 0x3F200000 as *mut u32;
            let level_reg = gpio_base.add(13 + N / 32);
            (level_reg.read_volatile() & (1 << (N % 32))) != 0
        }
    }
    
    fn set_direction(&mut self, direction: Direction) {
        unsafe {
            let gpio_base = 0x3F200000 as *mut u32;
            let func_reg = gpio_base.add(N / 10);
            let shift = (N % 10) * 3;
            let value = match direction {
                Direction::Input => 0b000,
                Direction::Output => 0b001,
            };
            
            let current = func_reg.read_volatile();
            let mask = !(0b111 << shift);
            func_reg.write_volatile((current & mask) | (value << shift));
        }
    }
}
```

## System Calls and Interrupts

### Interrupt Handling
```rust
use x86_64::structures::idt::{InterruptDescriptorTable, InterruptStackFrame};

lazy_static! {
    static ref IDT: InterruptDescriptorTable = {
        let mut idt = InterruptDescriptorTable::new();
        idt.breakpoint.set_handler_fn(breakpoint_handler);
        idt.double_fault.set_handler_fn(double_fault_handler);
        idt[InterruptIndex::Timer.as_usize()]
            .set_handler_fn(timer_interrupt_handler);
        idt
    };
}

extern "x86-interrupt" fn breakpoint_handler(stack_frame: InterruptStackFrame) {
    println!("EXCEPTION: BREAKPOINT\n{:#?}", stack_frame);
}

extern "x86-interrupt" fn double_fault_handler(
    stack_frame: InterruptStackFrame,
    _error_code: u64,
) -> ! {
    panic!("EXCEPTION: DOUBLE FAULT\n{:#?}", stack_frame);
}

extern "x86-interrupt" fn timer_interrupt_handler(_stack_frame: InterruptStackFrame) {
    print!(".");
    unsafe {
        PICS.lock().notify_end_of_interrupt(InterruptIndex::Timer.as_u8());
    }
}

// System call interface
#[no_mangle]
pub extern "C" fn syscall_handler(
    syscall_number: usize,
    arg1: usize,
    arg2: usize,
    arg3: usize,
) -> isize {
    match syscall_number {
        0 => sys_read(arg1, arg2 as *mut u8, arg3),
        1 => sys_write(arg1, arg2 as *const u8, arg3),
        2 => sys_open(arg1 as *const u8, arg2),
        60 => sys_exit(arg1 as i32),
        _ => -1,
    }
}

fn sys_write(fd: usize, buf: *const u8, count: usize) -> isize {
    if fd == 1 || fd == 2 {
        // stdout/stderr
        unsafe {
            let slice = core::slice::from_raw_parts(buf, count);
            for &byte in slice {
                serial_print!("{}", byte as char);
            }
        }
        count as isize
    } else {
        -1
    }
}
```

## Process Management

### Task Scheduling
```rust
use alloc::collections::VecDeque;

pub struct Task {
    id: TaskId,
    stack: Stack,
    state: TaskState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskState {
    Ready,
    Running,
    Blocked,
    Terminated,
}

pub struct Scheduler {
    ready_queue: VecDeque<TaskId>,
    current_task: Option<TaskId>,
    tasks: BTreeMap<TaskId, Task>,
}

impl Scheduler {
    pub fn new() -> Self {
        Self {
            ready_queue: VecDeque::new(),
            current_task: None,
            tasks: BTreeMap::new(),
        }
    }
    
    pub fn spawn_task(&mut self, task: Task) {
        let task_id = task.id;
        self.tasks.insert(task_id, task);
        self.ready_queue.push_back(task_id);
    }
    
    pub fn schedule(&mut self) -> Option<&mut Task> {
        if let Some(task_id) = self.ready_queue.pop_front() {
            self.current_task = Some(task_id);
            self.tasks.get_mut(&task_id)
        } else {
            None
        }
    }
    
    pub fn yield_current(&mut self) {
        if let Some(task_id) = self.current_task.take() {
            self.ready_queue.push_back(task_id);
        }
    }
}

// Context switching
#[repr(C)]
pub struct TaskContext {
    rsp: u64,
    r15: u64,
    r14: u64,
    r13: u64,
    r12: u64,
    rbp: u64,
    rbx: u64,
}

extern "C" {
    fn switch_context(old: *mut TaskContext, new: *const TaskContext);
}

pub fn context_switch(old_task: &mut Task, new_task: &Task) {
    unsafe {
        switch_context(&mut old_task.context, &new_task.context);
    }
}
```

## File Systems

### Basic VFS Implementation
```rust
use alloc::string::String;
use alloc::vec::Vec;

pub trait FileSystem {
    fn read(&self, path: &str) -> Result<Vec<u8>, FsError>;
    fn write(&mut self, path: &str, data: &[u8]) -> Result<(), FsError>;
    fn list_dir(&self, path: &str) -> Result<Vec<String>, FsError>;
    fn create_file(&mut self, path: &str) -> Result<(), FsError>;
    fn delete(&mut self, path: &str) -> Result<(), FsError>;
}

#[derive(Debug)]
pub enum FsError {
    NotFound,
    PermissionDenied,
    AlreadyExists,
    InvalidPath,
    IoError,
}

// Simple in-memory filesystem
pub struct MemoryFs {
    files: BTreeMap<String, Vec<u8>>,
}

impl MemoryFs {
    pub fn new() -> Self {
        Self {
            files: BTreeMap::new(),
        }
    }
}

impl FileSystem for MemoryFs {
    fn read(&self, path: &str) -> Result<Vec<u8>, FsError> {
        self.files.get(path)
            .map(|data| data.clone())
            .ok_or(FsError::NotFound)
    }
    
    fn write(&mut self, path: &str, data: &[u8]) -> Result<(), FsError> {
        self.files.insert(path.to_string(), data.to_vec());
        Ok(())
    }
    
    fn list_dir(&self, path: &str) -> Result<Vec<String>, FsError> {
        let prefix = if path.ends_with('/') {
            path.to_string()
        } else {
            format!("{}/", path)
        };
        
        let files: Vec<String> = self.files.keys()
            .filter(|k| k.starts_with(&prefix))
            .map(|k| k.clone())
            .collect();
        
        Ok(files)
    }
    
    fn create_file(&mut self, path: &str) -> Result<(), FsError> {
        if self.files.contains_key(path) {
            Err(FsError::AlreadyExists)
        } else {
            self.files.insert(path.to_string(), Vec::new());
            Ok(())
        }
    }
    
    fn delete(&mut self, path: &str) -> Result<(), FsError> {
        self.files.remove(path)
            .map(|_| ())
            .ok_or(FsError::NotFound)
    }
}
```

## Embedded Systems

### Bare Metal Programming
```rust
#![no_std]
#![no_main]

use cortex_m_rt::entry;
use panic_halt as _;

#[entry]
fn main() -> ! {
    let peripherals = init_peripherals();
    
    loop {
        // Main application loop
        handle_sensors(&peripherals);
        update_actuators(&peripherals);
        cortex_m::asm::wfi(); // Wait for interrupt
    }
}

fn init_peripherals() -> Peripherals {
    let mut peripherals = Peripherals::take().unwrap();
    
    // Configure clock
    let mut rcc = peripherals.RCC.constrain();
    let clocks = rcc.cfgr.sysclk(48.mhz()).freeze();
    
    // Configure GPIO
    let mut gpioa = peripherals.GPIOA.split(&mut rcc.ahb);
    let led = gpioa.pa5.into_push_pull_output(&mut gpioa.moder, &mut gpioa.otyper);
    
    Peripherals { led, clocks }
}

struct Peripherals {
    led: PA5<Output<PushPull>>,
    clocks: Clocks,
}

fn handle_sensors(peripherals: &Peripherals) {
    // Read sensor data
}

fn update_actuators(peripherals: &Peripherals) {
    // Update outputs
}

// Real-time constraints
use cortex_m::interrupt;

fn critical_section<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    interrupt::free(|_| f())
}
```

## Key Learning Points

1. **No-Std Programming**: Working without standard library
2. **Memory Safety**: Safe abstractions over unsafe hardware access
3. **Real-time Constraints**: Deterministic timing and interrupt handling
4. **Hardware Abstraction**: Clean APIs over low-level hardware
5. **Resource Management**: Efficient use of limited system resources

## Tools and Crates
- `cortex-m` - ARM Cortex-M specific functionality
- `embedded-hal` - Hardware abstraction layer traits
- `nb` - Non-blocking APIs
- `heapless` - Data structures without allocation
- `bootloader` - Bootloader for x86_64 kernels
- `x86_64` - x86_64 specific structures and functions

## Getting Started
1. Start with embedded projects using development boards
2. Learn `no_std` programming constraints
3. Understand memory-mapped I/O and registers
4. Practice interrupt-driven programming
5. Build simple device drivers
6. Experiment with real-time operating systems
