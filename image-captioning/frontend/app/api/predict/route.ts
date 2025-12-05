import { type NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const file = formData.get("file") as File

    if (!file) {
      return NextResponse.json({ error: "No file provided" }, { status: 400 })
    }
    // Forward the uploaded file to the real backend `/predict` endpoint.
    // Set `BACKEND_URL` in your environment (e.g. .env.local) to override default.
    const BACKEND_URL = process.env.BACKEND_URL || "http://127.0.0.1:8000"

    // Allow optionally sending `top_k` from the incoming form data
    const topK = formData.get("top_k")
    const backendUrl = `${BACKEND_URL.replace(/\/$/, "")}/predict${topK ? `?top_k=${encodeURIComponent(String(topK))}` : ""}`

    const backendFormData = new FormData()
    // Preserve filename if available
    const filename = (file as any).name || "upload"
    backendFormData.append("file", file, filename)

    console.log("Forwarding prediction to backend:", backendUrl)
    const response = await fetch(backendUrl, {
      method: "POST",
      body: backendFormData,
    })

    // Try to parse JSON, fallback to text for diagnostics
    let data: any = null
    const contentType = response.headers.get("content-type") || ""
    try {
      if (contentType.includes("application/json")) {
        data = await response.json()
      } else {
        data = { text: await response.text() }
      }
    } catch (e) {
      data = { error: `Failed to parse backend response: ${String(e)}` }
    }

    // If backend returned non-OK, include diagnostics in the response
    if (!response.ok) {
      console.error("Backend /predict returned non-OK:", response.status, data)
      return NextResponse.json(
        { error: "Backend error", status: response.status, details: data },
        { status: 502 }
      )
    }

    console.log("Backend responded with status", response.status)
    return NextResponse.json(data, { status: 200 })
  } catch (error) {
    console.error("Prediction error:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
