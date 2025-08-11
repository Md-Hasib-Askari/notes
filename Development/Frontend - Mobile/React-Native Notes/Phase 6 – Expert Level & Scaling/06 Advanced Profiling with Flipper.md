# Advanced Performance Profiling with Flipper

Setup
- Enable Flipper in debug builds (RN template already includes)
- Use Hermes engine for better profiling; enable in android/app/build.gradle and iOS Podfile

Key plugins
- React DevTools: inspect component tree and re-renders
- Performance: track JS frame rate, UI thread, and dropped frames
- Network: inspect API calls; look for n+1 patterns
- Databases: view SQLite/Realm content

Hermes profiling
- Use Hermes sampling profiler to capture JS CPU profiles
- Analyze hotspots, long tasks, and GC pauses

Workflow
1) Reproduce slow screen
2) Record performance session
3) Identify heavy components (large props or rerenders)
4) Add memoization, split lists, or move work off JS thread
5) Re-test; automate via perf budgets where possible

Native profiling
- Android Studio Profiler (CPU/memory/allocations)
- Xcode Instruments (Time Profiler, Allocations, Leaks)
