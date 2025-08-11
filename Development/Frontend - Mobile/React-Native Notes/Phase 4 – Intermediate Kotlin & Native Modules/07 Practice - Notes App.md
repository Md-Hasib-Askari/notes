# Practice – Notes App (Room) + RN Bridge

Goal
- Build a Kotlin Notes app with Room, then expose a small native feature to a React Native app.

Part A: Kotlin Notes App
Features
- Create, list, delete notes (title, body, createdAt)
- RecyclerView + ListAdapter; Room + ViewModel + Flow

Steps
1) Create project (Empty Views Activity, Kotlin)
2) Add Room dependencies; set up Entity/DAO/Database/Repository/ViewModel
3) Build UI with RecyclerView and a simple Add screen/dialog
4) Add basic theming and handle rotation

Part B: Native Module for RN
Feature example: expose notes count to RN

Steps
1) In a bare RN project, create a Kotlin module (DeviceInfoModule example → NotesModule)
2) Implement a method getNotesCount(promise) that queries Room database and resolves an Int
3) Register package and consume from JS via NativeModules. Display count in a header badge

Optional enhancements
- WorkManager: periodic backup to JSON in Files directory
- Retrofit: sync notes with a sample REST service
- DataStore: store user settings (theme, sort order)

Notes
- Keep Android and RN projects separate for clarity in learning
- For production, consider a single RN app with native modules under android/
