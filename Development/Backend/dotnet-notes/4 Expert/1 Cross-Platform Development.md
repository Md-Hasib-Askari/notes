# Cross-Platform Development (Expert Notes)

## 1. Blazor (WebAssembly for Client-Side Apps)

* **Blazor WebAssembly**:

  * Run C# directly in the browser via WebAssembly.
  * No JavaScript required, though JS interop is supported.
  * Uses Razor components for UI.

  ```razor
  <h3>@message</h3>
  <button @onclick="ChangeMessage">Click</button>

  @code {
      private string message = "Hello Blazor!";
      void ChangeMessage() => message = "Updated!";
  }
  ```
* **Blazor Server**:

  * UI updates managed over SignalR.
  * Smaller payload, faster initial load.

---

## 2. MAUI/Xamarin (Mobile Apps with .NET)

* **Xamarin.Forms**:

  * Older framework for cross-platform mobile apps.
  * Shared UI + platform-specific renderers.
* **.NET MAUI (Multi-platform App UI)**:

  * Successor to Xamarin.
  * Single project targeting Android, iOS, macOS, Windows.
  * Unified UI with platform optimizations.

  ```csharp
  public partial class MainPage : ContentPage
  {
      public MainPage()
      {
          InitializeComponent();
          label.Text = "Hello MAUI!";
      }
  }
  ```

---

## 3. WPF/WinUI (Desktop Apps)

* **WPF (Windows Presentation Foundation)**:

  * Rich desktop UI framework using XAML.
  * Data binding, MVVM support.
  * Great for enterprise apps.
* **WinUI**:

  * Latest UI framework for Windows desktop.
  * Used in Windows App SDK.
  * Fluent Design System, modern controls.
* **Cross-platform alternatives**:

  * Avalonia UI, Uno Platform (if Windows-only not acceptable).
