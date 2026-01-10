# P.A.T.C.H Backend API Documentation

## For Frontend Integration

> **Base URL:** `http://localhost:5000/v1`  
> **Authentication:** JWT Bearer Token (except for auth endpoints)

---

## 🔓 Authentication Endpoints

### All Platforms

Base: `/v1/auth`

#### Register User

```http
POST /v1/auth/register
Content-Type: application/json
```

**Request Body:**

```json
{
  "username": "john_doe",
  "email": "john@example.com",
  "password": "SecurePassword123!"
}
```

**Response (201 Created):**

```json
{
  "id": 1,
  "username": "john_doe",
  "email": "john@example.com",
  "is_active": true
}
```

**Errors:**

- `409 Conflict` - Username already exists
- `400 Bad Request` - Password exceeds 72 bytes

---

#### Login

```http
POST /v1/auth/login
Content-Type: application/x-www-form-urlencoded
```

**Request Body (form-urlencoded):**

```
username=john_doe&password=SecurePassword123!
```

**JavaScript Example:**

```javascript
const formData = new URLSearchParams();
formData.append("username", "john_doe");
formData.append("password", "SecurePassword123!");

const response = await fetch("http://localhost:5000/v1/auth/login", {
  method: "POST",
  headers: {
    "Content-Type": "application/x-www-form-urlencoded",
  },
  body: formData,
});

const data = await response.json();
localStorage.setItem("access_token", data.access_token);
```

**Response (200 OK):**

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

**Store the `access_token` and include it in all protected requests!**

---

#### Get Current User

```http
GET /v1/auth/me
Authorization: Bearer {access_token}
```

**Response (200 OK):**

```json
{
  "id": 1,
  "username": "john_doe",
  "email": "john@example.com",
  "is_active": true
}
```

---

## 🎭 Persona Management

### All Platforms

Base: `/v1/persona` (🔒 **Requires Auth**)

#### Create Persona

```http
POST /v1/persona
Authorization: Bearer {access_token}
Content-Type: application/json
```

**Request Body:**

```json
{
  "name": "Technical Expert",
  "description": "Specializes in software architecture and best practices",
  "traits": ["analytical", "detail-oriented", "pragmatic"],
  "goals": ["provide accurate technical guidance", "suggest best practices"]
}
```

**Response (201 Created):**

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Technical Expert",
  "description": "Specializes in software architecture and best practices",
  "traits": ["analytical", "detail-oriented", "pragmatic"],
  "goals": ["provide accurate technical guidance", "suggest best practices"]
}
```

---

#### Get All Personas

```http
GET /v1/persona
Authorization: Bearer {access_token}
```

**Response (200 OK):**

```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "Technical Expert",
    "description": "Specializes in software architecture",
    "traits": ["analytical", "detail-oriented"],
    "goals": ["provide guidance"]
  },
  {
    "id": "660e8400-e29b-41d4-a716-446655440001",
    "name": "Creative Assistant",
    "description": "Helps with creative writing",
    "traits": ["imaginative", "supportive"],
    "goals": ["inspire creativity"]
  }
]
```

---

#### Get Persona by ID

```http
GET /v1/persona/{persona_id}
Authorization: Bearer {access_token}
```

**Response (200 OK):**

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Technical Expert",
  "description": "Specializes in software architecture",
  "traits": ["analytical"],
  "goals": ["provide guidance"]
}
```

**Error:**

- `404 Not Found` - Persona doesn't exist

---

#### Update Persona

```http
PUT /v1/persona/{persona_id}
Authorization: Bearer {access_token}
Content-Type: application/json
```

**Request Body (all fields optional):**

```json
{
  "name": "Senior Technical Expert",
  "description": "Updated description",
  "traits": ["analytical", "experienced"],
  "goals": ["mentor developers"]
}
```

**Response (200 OK):**

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Senior Technical Expert",
  "description": "Updated description",
  "traits": ["analytical", "experienced"],
  "goals": ["mentor developers"]
}
```

---

#### Delete Persona

```http
DELETE /v1/persona/{persona_id}
Authorization: Bearer {access_token}
```

**Response:** `204 No Content`

---

## 🧠 Context Management (Redis)

### All Platforms

Base: `/v1/context` (🔒 **Requires Auth**)

#### Update/Save User Context

```http
PUT /v1/context/{user_id}
Authorization: Bearer {access_token}
Content-Type: application/json
```

**Request Body:**

```json
{
  "context_data": {
    "current_project": "E-commerce Platform",
    "programming_language": "Python",
    "framework": "FastAPI",
    "preferences": {
      "code_style": "PEP8",
      "documentation": "detailed"
    }
  }
}
```

**Response (200 OK):**

```json
{
  "user_id": "user_12345",
  "context_data": {
    "current_project": "E-commerce Platform",
    "programming_language": "Python",
    "framework": "FastAPI",
    "preferences": {
      "code_style": "PEP8",
      "documentation": "detailed"
    }
  },
  "updated_at": "2026-01-08T14:30:00Z"
}
```

---

#### Get User Context

```http
GET /v1/context/{user_id}
Authorization: Bearer {access_token}
```

**Response (200 OK):**

```json
{
  "user_id": "user_12345",
  "context_data": {
    "current_project": "E-commerce Platform",
    "programming_language": "Python"
  },
  "updated_at": "2026-01-08T14:30:00Z"
}
```

**If no context exists:**

```json
{
  "user_id": "user_12345",
  "context_data": {},
  "updated_at": "N/A"
}
```

---

#### Delete User Context

```http
DELETE /v1/context/{user_id}
Authorization: Bearer {access_token}
```

**Response:** `204 No Content`

**Error:**

- `404 Not Found` - No context exists for this user

---

## 📄 Document Management (ChromaDB)

### All Platforms

Base: `/v1/documents` (🔒 **Requires Auth**)

#### Create Collection

```http
POST /v1/documents/collections
Authorization: Bearer {access_token}
Content-Type: application/json
```

**Request Body:**

```json
{
  "name": "company_policies",
  "metadata": {
    "description": "Company policy documents",
    "created_by": "admin"
  }
}
```

**Response (201 Created):**

```json
{
  "name": "company_policies",
  "metadata": {
    "description": "Company policy documents",
    "created_by": "admin"
  }
}
```

---

#### Add Documents to Collection

```http
POST /v1/documents/{collection_name}/documents
Authorization: Bearer {access_token}
Content-Type: application/json
```

**Request Body:**

```json
{
  "documents": [
    {
      "id": "doc_001",
      "content": "This is the content of the first document about our refund policy...",
      "metadata": {
        "category": "policies",
        "department": "customer_service",
        "last_updated": "2026-01-01"
      }
    },
    {
      "id": "doc_002",
      "content": "Employee handbook section on vacation days and time off...",
      "metadata": {
        "category": "hr",
        "department": "human_resources"
      }
    }
  ]
}
```

**Response (201 Created):**

```json
{
  "collection_name": "company_policies",
  "added_count": 2,
  "ids": ["doc_001", "doc_002"]
}
```

---

#### Query Documents (Semantic Search)

```http
POST /v1/documents/{collection_name}/query
Authorization: Bearer {access_token}
Content-Type: application/json
```

**Request Body:**

```json
{
  "query_texts": ["What is the refund policy?"],
  "n_results": 5,
  "where": {
    "category": "policies"
  }
}
```

**Response (200 OK):**

```json
[
  [
    {
      "id": "doc_001",
      "content": "This is the content of the first document about our refund policy...",
      "metadata": {
        "category": "policies",
        "department": "customer_service"
      },
      "distance": 0.234
    },
    {
      "id": "doc_003",
      "content": "Related policy document...",
      "metadata": {
        "category": "policies"
      },
      "distance": 0.456
    }
  ]
]
```

**Note:** Lower `distance` = higher similarity

---

#### Delete Collection

```http
DELETE /v1/documents/collections/{collection_name}
Authorization: Bearer {access_token}
```

**Response:** `204 No Content`

---

#### Delete Documents from Collection

```http
DELETE /v1/documents/{collection_name}/documents
Authorization: Bearer {access_token}
Content-Type: application/json
```

**Request Body (provide either `ids` or `where`):**

```json
{
  "ids": ["doc_001", "doc_002"]
}
```

**Or filter by metadata:**

```json
{
  "where": {
    "category": "outdated"
  }
}
```

**Response:** `204 No Content`

**Error:**

- `400 Bad Request` - Neither `ids` nor `where` provided

---

## 💬 AI Chat (RAG-Powered)

### All Platforms

Base: `/v1/chat` (🔒 **Requires Auth**)

#### Send Chat Message

```http
POST /v1/chat
Authorization: Bearer {access_token}
Content-Type: application/json
```

**Request Body:**

```json
{
  "user_message": "What are the key features of FastAPI?",
  "collection_name": "technical_docs",
  "user_id": "user_12345",
  "past_messages": [
    {
      "role": "user",
      "content": "Tell me about Python frameworks"
    },
    {
      "role": "model",
      "content": "Python has several popular frameworks like Django, FastAPI, Flask..."
    }
  ]
}
```

**Fields:**

- `user_message` (required): Current user question
- `collection_name` (required): ChromaDB collection to query for context
- `user_id` (required): Unique identifier for the user/session
- `past_messages` (optional): Recent conversation history (array of messages)

**Response (200 OK):**

```json
{
  "ai_response": "FastAPI is a modern, fast (high-performance) web framework for building APIs with Python 3.7+ based on standard Python type hints. Key features include:\n\n1. **Fast**: Very high performance, on par with NodeJS and Go\n2. **Type Safety**: Automatic validation using Pydantic\n3. **Auto Documentation**: Interactive API docs (Swagger UI)\n4. **Async Support**: Built-in support for async/await\n5. **Easy to Learn**: Simple and intuitive syntax",
  "source_documents": [
    {
      "id": "fastapi_intro",
      "content": "FastAPI is a modern, fast web framework...",
      "metadata": {
        "category": "web_frameworks",
        "language": "Python"
      },
      "distance": 0.123
    },
    {
      "id": "fastapi_features",
      "content": "Key features of FastAPI include...",
      "metadata": {
        "category": "documentation"
      },
      "distance": 0.234
    }
  ],
  "message_id": "msg_7a3f9e2b-4c1d-4f8e-9b2a-6d5c8e1f3a4b"
}
```

---

## ❤️ Health Check

### All Platforms

Base: `/v1/health` (✅ **Public - No Auth Required**)

#### Check API Health

```http
GET /v1/health
```

**Response (200 OK):**

```
Hello world
```

---

## 🔑 Authentication Helper

### JavaScript/TypeScript Example

```javascript
class APIClient {
  constructor(baseURL = "http://localhost:5000/v1") {
    this.baseURL = baseURL;
    this.token = localStorage.getItem("access_token");
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const headers = {
      "Content-Type": "application/json",
      ...options.headers,
    };

    // Add auth header for protected routes
    if (
      this.token &&
      !endpoint.includes("/auth/login") &&
      !endpoint.includes("/auth/register")
    ) {
      headers["Authorization"] = `Bearer ${this.token}`;
    }

    const config = {
      ...options,
      headers,
    };

    const response = await fetch(url, config);

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Request failed");
    }

    // Handle 204 No Content
    if (response.status === 204) {
      return null;
    }

    return response.json();
  }

  // Auth methods
  async login(username, password) {
    const formData = new URLSearchParams();
    formData.append("username", username);
    formData.append("password", password);

    const data = await this.request("/auth/login", {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body: formData,
    });

    this.token = data.access_token;
    localStorage.setItem("access_token", data.access_token);
    return data;
  }

  async register(username, email, password) {
    return this.request("/auth/register", {
      method: "POST",
      body: JSON.stringify({ username, email, password }),
    });
  }

  // Persona methods
  async createPersona(personaData) {
    return this.request("/persona", {
      method: "POST",
      body: JSON.stringify(personaData),
    });
  }

  async getAllPersonas() {
    return this.request("/persona");
  }

  // Chat method
  async sendChatMessage(
    userMessage,
    collectionName,
    userId,
    pastMessages = []
  ) {
    return this.request("/chat", {
      method: "POST",
      body: JSON.stringify({
        user_message: userMessage,
        collection_name: collectionName,
        user_id: userId,
        past_messages: pastMessages,
      }),
    });
  }
}

// Usage
const api = new APIClient();

// Login
await api.login("john_doe", "password123");

// Create persona
const persona = await api.createPersona({
  name: "Helper",
  description: "Helpful assistant",
  traits: ["friendly"],
  goals: ["help users"],
});

// Send chat message
const response = await api.sendChatMessage(
  "What is FastAPI?",
  "technical_docs",
  "user_123"
);
console.log(response.ai_response);
```

---

## 📋 Quick Reference

| Endpoint                           | Method | Auth | Description       |
| ---------------------------------- | ------ | ---- | ----------------- |
| `/v1/auth/register`                | POST   | ❌   | Register new user |
| `/v1/auth/login`                   | POST   | ❌   | Login & get token |
| `/v1/auth/me`                      | GET    | ✅   | Get current user  |
| `/v1/persona`                      | POST   | ✅   | Create persona    |
| `/v1/persona`                      | GET    | ✅   | List all personas |
| `/v1/persona/{id}`                 | GET    | ✅   | Get persona by ID |
| `/v1/persona/{id}`                 | PUT    | ✅   | Update persona    |
| `/v1/persona/{id}`                 | DELETE | ✅   | Delete persona    |
| `/v1/context/{user_id}`            | PUT    | ✅   | Save context      |
| `/v1/context/{user_id}`            | GET    | ✅   | Get context       |
| `/v1/context/{user_id}`            | DELETE | ✅   | Delete context    |
| `/v1/documents/collections`        | POST   | ✅   | Create collection |
| `/v1/documents/{name}/documents`   | POST   | ✅   | Add documents     |
| `/v1/documents/{name}/query`       | POST   | ✅   | Search documents  |
| `/v1/documents/collections/{name}` | DELETE | ✅   | Delete collection |
| `/v1/chat`                         | POST   | ✅   | Send chat message |
| `/v1/health`                       | GET    | ❌   | Health check      |

---

## 🚨 Error Responses

All endpoints return errors in this format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

**Common HTTP Status Codes:**

- `200 OK` - Success
- `201 Created` - Resource created
- `204 No Content` - Success with no response body
- `400 Bad Request` - Invalid request data
- `401 Unauthorized` - Missing or invalid token
- `404 Not Found` - Resource doesn't exist
- `409 Conflict` - Resource already exists
- `500 Internal Server Error` - Server error

---

## 🔧 Environment Setup

**Required Environment Variables:**

```bash
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/cogniflowdb
REDIS_HOST=localhost
REDIS_PORT=6379
CHROMADB_HOST=localhost
CHROMADB_PORT=8000
GEMINI_API_KEY=your_google_gemini_api_key
SECRET_KEY=your_jwt_secret_key
```

---

## 📞 Support

For issues or questions:

- Check Swagger docs at: `http://localhost:5000/docs`
- Interactive API at: `http://localhost:5000/redoc`
