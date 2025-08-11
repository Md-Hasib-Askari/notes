# Retrofit for REST APIs

Add dependencies
- retrofit2, OkHttp, logging-interceptor
- Converter: kotlinx-serialization or Moshi

Models (kotlinx.serialization)
```kotlin
@Serializable data class Post(val id: Int, val title: String, val body: String)
```

API interface
```kotlin
interface Api {
    @GET("posts") suspend fun posts(): List<Post>
    @GET("posts/{id}") suspend fun post(@Path("id") id: Int): Post
}
```

Retrofit builder
```kotlin
val client = OkHttpClient.Builder()
    .addInterceptor(HttpLoggingInterceptor().apply { level = HttpLoggingInterceptor.Level.BODY })
    .build()

val retrofit = Retrofit.Builder()
    .baseUrl("https://jsonplaceholder.typicode.com/")
    .addConverterFactory(Json.asConverterFactory("application/json".toMediaType()))
    .client(client)
    .build()

val api = retrofit.create(Api::class.java)
```

Usage
```kotlin
viewModelScope.launch {
    try {
        val list = api.posts()
        // update UI via LiveData/StateFlow
    } catch (e: IOException) {
        // network error
    } catch (e: HttpException) {
        // non-2xx
    }
}
```

Tips
- Use Result wrappers and map errors to UI
- Add timeouts and retry/backoff when appropriate
- Cache with OkHttp or persist results to Room
