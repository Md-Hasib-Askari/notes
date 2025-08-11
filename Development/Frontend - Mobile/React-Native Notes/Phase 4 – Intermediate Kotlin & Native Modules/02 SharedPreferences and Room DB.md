# SharedPreferences & Room DB

SharedPreferences (quick key-value)
```kotlin
val prefs = getSharedPreferences("app", MODE_PRIVATE)
prefs.edit().putBoolean("dark", true).apply()
val dark = prefs.getBoolean("dark", false)
```

Note: Prefer Jetpack DataStore for modern reactive storage. Use SharedPreferences for very small flags.

Room setup
- Add Room dependencies in Gradle
- Create Entity, DAO, Database; use coroutines and Flow

Entity and DAO
```kotlin
@Entity(tableName = "notes")
data class NoteEntity(
    @PrimaryKey val id: String,
    val title: String,
    val body: String,
    val createdAt: Long = System.currentTimeMillis()
)

@Dao
interface NoteDao {
    @Query("SELECT * FROM notes ORDER BY createdAt DESC")
    fun watchAll(): Flow<List<NoteEntity>>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsert(note: NoteEntity)

    @Query("DELETE FROM notes WHERE id = :id")
    suspend fun delete(id: String)
}
```

Database
```kotlin
@Database(entities = [NoteEntity::class], version = 1)
abstract class AppDb : RoomDatabase() { abstract fun notes(): NoteDao }

fun provideDb(context: Context) = Room.databaseBuilder(context, AppDb::class.java, "notes.db").build()
```

Repository + ViewModel
```kotlin
class NotesRepo(private val dao: NoteDao) {
    val notes: Flow<List<NoteEntity>> = dao.watchAll()
    suspend fun add(title: String, body: String) = dao.upsert(NoteEntity(UUID.randomUUID().toString(), title, body))
    suspend fun remove(id: String) = dao.delete(id)
}

class NotesVm(app: Application) : AndroidViewModel(app) {
    private val db = provideDb(app)
    private val repo = NotesRepo(db.notes())
    val notes = repo.notes.asLiveData()
    fun add(title: String, body: String) = viewModelScope.launch { repo.add(title, body) }
}
```

Observing in Activity
```kotlin
vm.notes.observe(this) { list -> adapter.submitList(list.map { NoteUi(it.id, it.title, it.body) }) }
```

Migrations
- Bump Database version when schema changes
- Provide migration scripts to keep user data
