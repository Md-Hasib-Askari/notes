# NestJS Phase 1: Hands-on Practice Guide

## 1. Create Your First NestJS Application

**Step-by-Step Setup:**
```bash
# Install NestJS CLI globally
npm install -g @nestjs/cli

# Create new application
nest new my-first-app

# Navigate to project
cd my-first-app

# Start development server
npm run start:dev
```

**Verify Installation:**
- Open browser: `http://localhost:3000`
- Should see "Hello World!" message
- Check file structure in `src/` folder

## 2. Build a Simple REST API with CRUD Operations

### Create a Books API

**Generate Resources:**
```bash
nest g module books
nest g controller books
nest g service books
```

**Book Entity (books/entities/book.entity.ts):**
```typescript
export class Book {
  id: string;
  title: string;
  author: string;
  isbn: string;
  publishedYear: number;
  genre: string;
}
```

**Data Transfer Objects:**
```typescript
// books/dto/create-book.dto.ts
export class CreateBookDto {
  title: string;
  author: string;
  isbn: string;
  publishedYear: number;
  genre: string;
}

// books/dto/update-book.dto.ts
export class UpdateBookDto {
  title?: string;
  author?: string;
  isbn?: string;
  publishedYear?: number;
  genre?: string;
}
```

**Books Service (books/books.service.ts):**
```typescript
@Injectable()
export class BooksService {
  private books: Book[] = [
    {
      id: '1',
      title: 'The Great Gatsby',
      author: 'F. Scott Fitzgerald',
      isbn: '978-0-7432-7356-5',
      publishedYear: 1925,
      genre: 'Fiction'
    }
  ];

  findAll(): Book[] {
    return this.books;
  }

  findOne(id: string): Book {
    const book = this.books.find(book => book.id === id);
    if (!book) {
      throw new NotFoundException(`Book with ID ${id} not found`);
    }
    return book;
  }

  create(createBookDto: CreateBookDto): Book {
    const newBook: Book = {
      id: Date.now().toString(),
      ...createBookDto
    };
    this.books.push(newBook);
    return newBook;
  }

  update(id: string, updateBookDto: UpdateBookDto): Book {
    const bookIndex = this.books.findIndex(book => book.id === id);
    if (bookIndex === -1) {
      throw new NotFoundException(`Book with ID ${id} not found`);
    }
    this.books[bookIndex] = { ...this.books[bookIndex], ...updateBookDto };
    return this.books[bookIndex];
  }

  remove(id: string): void {
    const bookIndex = this.books.findIndex(book => book.id === id);
    if (bookIndex === -1) {
      throw new NotFoundException(`Book with ID ${id} not found`);
    }
    this.books.splice(bookIndex, 1);
  }
}
```

## 3. Implement Basic Routing and HTTP Methods

**Books Controller (books/books.controller.ts):**
```typescript
@Controller('books')
export class BooksController {
  constructor(private readonly booksService: BooksService) {}

  // GET /books - Get all books
  @Get()
  findAll(): Book[] {
    return this.booksService.findAll();
  }

  // GET /books/:id - Get book by ID
  @Get(':id')
  findOne(@Param('id') id: string): Book {
    return this.booksService.findOne(id);
  }

  // POST /books - Create new book
  @Post()
  create(@Body() createBookDto: CreateBookDto): Book {
    return this.booksService.create(createBookDto);
  }

  // PUT /books/:id - Update book
  @Put(':id')
  update(
    @Param('id') id: string,
    @Body() updateBookDto: UpdateBookDto
  ): Book {
    return this.booksService.update(id, updateBookDto);
  }

  // DELETE /books/:id - Delete book
  @Delete(':id')
  remove(@Param('id') id: string): void {
    return this.booksService.remove(id);
  }
}
```

## 4. Work with Request/Response Objects

**Advanced Controller Examples:**
```typescript
@Controller('books')
export class BooksController {
  constructor(private readonly booksService: BooksService) {}

  // Working with Response object
  @Post()
  create(
    @Body() createBookDto: CreateBookDto,
    @Res() res: Response
  ): Response {
    try {
      const book = this.booksService.create(createBookDto);
      return res.status(201).json({
        success: true,
        message: 'Book created successfully',
        data: book
      });
    } catch (error) {
      return res.status(400).json({
        success: false,
        message: 'Failed to create book',
        error: error.message
      });
    }
  }

  // Working with Request object
  @Get('search')
  search(@Req() req: Request): Book[] {
    const userAgent = req.headers['user-agent'];
    const clientIp = req.ip;
    console.log(`Search request from ${clientIp} using ${userAgent}`);
    
    return this.booksService.findAll();
  }

  // Custom headers
  @Get(':id')
  findOne(
    @Param('id') id: string,
    @Headers('authorization') auth: string
  ): Book {
    console.log('Authorization header:', auth);
    return this.booksService.findOne(id);
  }
}
```

## 5. Handle Query Parameters and Route Parameters

**Query Parameters Examples:**
```typescript
@Controller('books')
export class BooksController {
  // GET /books?page=1&limit=10&author=tolkien&genre=fantasy
  @Get()
  findAll(
    @Query('page') page: number = 1,
    @Query('limit') limit: number = 10,
    @Query('author') author?: string,
    @Query('genre') genre?: string,
    @Query('year') publishedYear?: number
  ) {
    return this.booksService.findAllWithFilters({
      page: Number(page),
      limit: Number(limit),
      author,
      genre,
      publishedYear: publishedYear ? Number(publishedYear) : undefined
    });
  }

  // GET /books/search?q=gatsby&sort=title&order=asc
  @Get('search')
  search(
    @Query('q') query: string,
    @Query('sort') sortBy: string = 'title',
    @Query('order') order: 'asc' | 'desc' = 'asc'
  ) {
    return this.booksService.search(query, sortBy, order);
  }

  // Route parameters with validation
  @Get('author/:authorName/books/:bookId')
  findBookByAuthor(
    @Param('authorName') authorName: string,
    @Param('bookId') bookId: string
  ) {
    return this.booksService.findBookByAuthor(authorName, bookId);
  }

  // Multiple route parameters
  @Get('category/:genre/year/:year')
  findByGenreAndYear(
    @Param('genre') genre: string,
    @Param('year') year: string
  ) {
    return this.booksService.findByGenreAndYear(genre, Number(year));
  }
}
```

**Enhanced Service with Filtering:**
```typescript
@Injectable()
export class BooksService {
  // ...existing code...

  findAllWithFilters(filters: {
    page: number;
    limit: number;
    author?: string;
    genre?: string;
    publishedYear?: number;
  }): { books: Book[]; total: number; page: number; limit: number } {
    let filteredBooks = [...this.books];

    // Apply filters
    if (filters.author) {
      filteredBooks = filteredBooks.filter(book =>
        book.author.toLowerCase().includes(filters.author.toLowerCase())
      );
    }

    if (filters.genre) {
      filteredBooks = filteredBooks.filter(book =>
        book.genre.toLowerCase() === filters.genre.toLowerCase()
      );
    }

    if (filters.publishedYear) {
      filteredBooks = filteredBooks.filter(book =>
        book.publishedYear === filters.publishedYear
      );
    }

    // Pagination
    const startIndex = (filters.page - 1) * filters.limit;
    const endIndex = startIndex + filters.limit;
    const paginatedBooks = filteredBooks.slice(startIndex, endIndex);

    return {
      books: paginatedBooks,
      total: filteredBooks.length,
      page: filters.page,
      limit: filters.limit
    };
  }

  search(query: string, sortBy: string, order: 'asc' | 'desc'): Book[] {
    const results = this.books.filter(book =>
      book.title.toLowerCase().includes(query.toLowerCase()) ||
      book.author.toLowerCase().includes(query.toLowerCase())
    );

    return results.sort((a, b) => {
      const aValue = a[sortBy as keyof Book];
      const bValue = b[sortBy as keyof Book];
      
      if (order === 'asc') {
        return aValue > bValue ? 1 : -1;
      } else {
        return aValue < bValue ? 1 : -1;
      }
    });
  }
}
```

## Testing Your API

**Using curl commands:**
```bash
# Get all books
curl http://localhost:3000/books

# Get book by ID
curl http://localhost:3000/books/1

# Create new book
curl -X POST http://localhost:3000/books \
  -H "Content-Type: application/json" \
  -d '{"title":"1984","author":"George Orwell","isbn":"978-0-452-28423-4","publishedYear":1949,"genre":"Dystopian"}'

# Update book
curl -X PUT http://localhost:3000/books/1 \
  -H "Content-Type: application/json" \
  -d '{"title":"The Great Gatsby - Updated"}'

# Delete book
curl -X DELETE http://localhost:3000/books/1

# Search with query parameters
curl "http://localhost:3000/books?page=1&limit=5&author=orwell&genre=dystopian"
```

## Practice Exercises

1. **Add more endpoints**: Create endpoints for different book categories
2. **Implement sorting**: Add sorting functionality to your API
3. **Add validation**: Use class-validator for input validation
4. **Error handling**: Implement comprehensive error responses
5. **Logging**: Add request logging to track API usage

This hands-on guide provides practical examples for building your first NestJS REST API with CRUD operations, proper routing, and parameter handling.
