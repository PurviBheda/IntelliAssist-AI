/**
 * IntelliAssist AI - Frontend API Client
 */

export interface Citation {
  chunk_id?: string
  filename?: string
  page_number?: number
  similarity_score?: number
  file_type?: string
  chunk_index?: number
  is_cited?: boolean
}

export interface DocumentRecord {
  document_id: string
  conversation_id?: string | null
  filename: string
  file_type: string
  total_pages: number
  total_characters: number
  chunk_count: number
  file_path?: string
  created_at?: string
}

export interface ChatMessage {
  message_id?: string
  conversation_id?: string
  role: "user" | "assistant"
  content: string
  created_at?: string
  citations?: Citation[]
  metadata?: Record<string, any>
  status?: "sending" | "success" | "error"
  error?: string
}

export interface ConversationItem {
  conversation_id: string
  title: string
  created_at: string
  updated_at: string
  message_count: number
  documents?: DocumentRecord[]
  messages?: ChatMessage[]
}

export interface ChatResponse {
  conversation_id: string
  user_message: string
  assistant_response: string
  citations: Citation[]
  retrieved_chunks?: any[]
  sufficient_context: boolean
  status: "success" | "insufficient_context" | "error"
  model_name?: string
  is_live_llm: boolean
  created_at: string
  error?: string | null
  intent?: string | null
  intent_confidence?: number | null
}

export interface UploadResponse {
  status: "success" | "error"
  message: string
  document: DocumentRecord
}

const BACKEND_DIRECT_URL = "http://127.0.0.1:8000"

async function requestWithFallback(
  path: string,
  options: RequestInit = {},
): Promise<Response> {
  // Try relative path first (via Vite proxy)
  try {
    const res = await fetch(path, options)
    // If not a 404 or connection error, return
    if (res.status !== 404 && res.status !== 502 && res.status !== 504) {
      return res
    }
  } catch (err) {
    // Vite proxy might not be forwarding, proceed to direct URL
  }

  // Fallback to direct backend URL
  const directUrl = `${BACKEND_DIRECT_URL}${
    path.startsWith("/") ? path : `/${path}`
  }`
  return fetch(directUrl, options)
}

export async function fetchHealth(): Promise<{
  status: string
  is_live_llm: boolean
  model_name: string
}> {
  const res = await requestWithFallback("/api/health")
  if (!res.ok) {
    throw new Error(`Health check failed: ${res.statusText}`)
  }
  return res.json()
}

export async function fetchConversations(
  limit = 50,
): Promise<ConversationItem[]> {
  const res = await requestWithFallback(`/api/conversations?limit=${limit}`)
  if (!res.ok) {
    throw new Error(`Failed to list conversations: ${res.statusText}`)
  }
  return res.json()
}

export async function fetchConversationDetails(
  conversationId: string,
): Promise<ConversationItem> {
  const res = await requestWithFallback(
    `/api/conversations/${encodeURIComponent(conversationId)}`,
  )
  if (!res.ok) {
    throw new Error(`Failed to load conversation: ${res.statusText}`)
  }
  return res.json()
}

export async function createConversation(
  title?: string,
  conversationId?: string,
): Promise<ConversationItem> {
  const res = await requestWithFallback("/api/conversations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title, conversation_id: conversationId }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(
      err.detail || `Failed to create conversation: ${res.statusText}`,
    )
  }
  return res.json()
}

export async function deleteConversation(
  conversationId: string,
): Promise<void> {
  const res = await requestWithFallback(
    `/api/conversations/${encodeURIComponent(conversationId)}`,
    {
      method: "DELETE",
    },
  )
  if (!res.ok) {
    throw new Error(`Failed to delete conversation: ${res.statusText}`)
  }
}

export async function uploadDocument(
  file: File,
  conversationId?: string,
): Promise<UploadResponse> {
  const formData = new FormData()
  formData.append("file", file)
  if (conversationId) {
    formData.append("conversation_id", conversationId)
  }

  const res = await requestWithFallback("/api/upload", {
    method: "POST",
    body: formData,
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Upload failed: ${res.statusText}`)
  }

  return res.json()
}

export async function removeDocument(
  conversationId: string,
  documentId: string,
): Promise<void> {
  const res = await requestWithFallback(
    `/api/conversations/${encodeURIComponent(conversationId)}/documents/${encodeURIComponent(documentId)}`,
    {
      method: "DELETE",
    },
  )
  if (!res.ok) {
    throw new Error(`Failed to remove document: ${res.statusText}`)
  }
}

export async function sendChatMessage(
  conversationId: string,
  message: string,
  autoCreate = true,
): Promise<ChatResponse> {
  const res = await requestWithFallback("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      conversation_id: conversationId,
      message,
      auto_create_conversation: autoCreate,
    }),
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Failed to send message: ${res.statusText}`)
  }

  return res.json()
}
