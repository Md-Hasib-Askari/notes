# RecyclerView & Adapters

Use ListAdapter + DiffUtil for efficient lists.

Item layout (item_note.xml)
```xml
<?xml version="1.0" encoding="utf-8"?>
<LinearLayout xmlns:android="http://schemas.android.com/apk/res/android"
    android:layout_width="match_parent"
    android:layout_height="wrap_content"
    android:orientation="vertical"
    android:padding="12dp">
    <TextView android:id="@+id/tvTitle" android:layout_width="match_parent" android:layout_height="wrap_content" android:textStyle="bold" android:textSize="16sp"/>
    <TextView android:id="@+id/tvBody" android:layout_width="match_parent" android:layout_height="wrap_content" android:maxLines="2" android:ellipsize="end"/>
</LinearLayout>
```

Adapter with DiffUtil
```kotlin
data class NoteUi(val id: String, val title: String, val body: String)

class NoteDiff : DiffUtil.ItemCallback<NoteUi>() {
    override fun areItemsTheSame(old: NoteUi, new: NoteUi) = old.id == new.id
    override fun areContentsTheSame(old: NoteUi, new: NoteUi) = old == new
}

class NoteAdapter(private val onClick: (NoteUi) -> Unit) : ListAdapter<NoteUi, NoteAdapter.VH>(NoteDiff()) {
    inner class VH(val binding: ItemNoteBinding): RecyclerView.ViewHolder(binding.root) {
        fun bind(item: NoteUi) = with(binding) {
            tvTitle.text = item.title
            tvBody.text = item.body
            root.setOnClickListener { onClick(item) }
        }
    }
    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): VH {
        val binding = ItemNoteBinding.inflate(LayoutInflater.from(parent.context), parent, false)
        return VH(binding)
    }
    override fun onBindViewHolder(holder: VH, position: Int) = holder.bind(getItem(position))
}
```

Attach to RecyclerView
```kotlin
val adapter = NoteAdapter { note -> /* navigate */ }
recyclerView.layoutManager = LinearLayoutManager(this)
recyclerView.adapter = adapter
adapter.submitList(notes)
```

Tips
- Use ItemDecoration for dividers/margins
- Use stableIds for animations on reorders
- Use Paging 3 for large datasets
