"use client"

import type React from "react"
import { useState, useRef } from "react"
import { Upload, Loader2, ImageIcon, X, Sparkles, MessageSquareText, Camera } from "lucide-react"
import { Button } from "@/components/ui/button"

const API_URL = "/api/predict"

interface PredictionResult {
  caption: string
  confidence?: number
}

export default function ImageCaptioning() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      if (!file.type.startsWith("image/")) {
        setError("Vui lòng chọn file ảnh")
        return
      }
      setSelectedImage(file)
      setPreview(URL.createObjectURL(file))
      setResult(null)
      setError(null)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files?.[0]
    if (file && file.type.startsWith("image/")) {
      setSelectedImage(file)
      setPreview(URL.createObjectURL(file))
      setResult(null)
      setError(null)
    }
  }

  const handlePredict = async () => {
    if (!selectedImage) return

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const formData = new FormData()
      formData.append("file", selectedImage)

      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      })

      if (!response.ok) throw new Error("Lỗi khi gọi API")

      const data = await response.json()
      setResult({
        caption: data.caption || data.description || data.text,
        confidence: data.confidence || data.probability,
      })
    } catch (err) {
      setError("Không thể kết nối đến server. Vui lòng thử lại.")
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const clearImage = () => {
    setSelectedImage(null)
    setPreview(null)
    setResult(null)
    setError(null)
    if (fileInputRef.current) fileInputRef.current.value = ""
  }

  return (
    <main className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-primary flex items-center justify-center">
              <MessageSquareText className="w-5 h-5 text-primary-foreground" />
            </div>
            <span className="text-xl font-semibold text-foreground font-serif">ImageCaption AI</span>
          </div>
          <nav className="hidden md:flex items-center gap-8 text-sm text-muted-foreground">
            <a href="#" className="hover:text-foreground transition-colors">
              Trang chủ
            </a>
            <a href="#" className="hover:text-foreground transition-colors">
              Ví dụ
            </a>
            <a href="#" className="hover:text-foreground transition-colors">
              Hướng dẫn
            </a>
          </nav>
        </div>
      </header>

      {/* Hero Section */}
      <section className="py-12 md:py-16 px-6">
        <div className="max-w-7xl mx-auto text-center mb-10 md:mb-12">
          <h1 className="text-3xl md:text-4xl lg:text-5xl font-bold text-foreground mb-4 font-serif text-balance">
            Mô tả ảnh tự động bằng AI
          </h1>
          <p className="text-base md:text-lg text-muted-foreground max-w-2xl mx-auto text-pretty">
            Tải lên bất kỳ hình ảnh nào và để AI tạo mô tả chi tiết cho bạn trong tích tắc.
          </p>
        </div>

        <div className="max-w-6xl mx-auto grid lg:grid-cols-2 gap-6 md:gap-8 items-start">
          {/* Left: Upload Section */}
          <div className="bg-card rounded-2xl border border-border shadow-sm overflow-hidden">
            <div className="p-5 md:p-6 border-b border-border">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-accent flex items-center justify-center">
                  <Camera className="w-4 h-4 text-accent-foreground" />
                </div>
                <div>
                  <h2 className="font-semibold text-foreground">Tải ảnh lên</h2>
                  <p className="text-sm text-muted-foreground">Hỗ trợ JPG, PNG, WEBP</p>
                </div>
              </div>
            </div>

            <div className="p-5 md:p-6">
              {!preview ? (
                <div
                  className={`relative border-2 border-dashed rounded-xl p-8 md:p-12 transition-all cursor-pointer
                    ${
                      isDragging
                        ? "border-primary bg-accent"
                        : "border-border hover:border-primary/50 hover:bg-secondary/50"
                    }`}
                  onClick={() => fileInputRef.current?.click()}
                  onDrop={handleDrop}
                  onDragOver={(e) => {
                    e.preventDefault()
                    setIsDragging(true)
                  }}
                  onDragLeave={() => setIsDragging(false)}
                >
                  <div className="flex flex-col items-center text-center">
                    <div className="w-16 h-16 rounded-full bg-secondary flex items-center justify-center mb-4">
                      <ImageIcon className="w-7 h-7 text-muted-foreground" />
                    </div>
                    <p className="font-medium text-foreground mb-1">Kéo thả ảnh vào đây</p>
                    <p className="text-sm text-muted-foreground mb-4">hoặc click để chọn file</p>
                    <Button variant="outline" size="sm" className="pointer-events-none bg-transparent">
                      <Upload className="w-4 h-4 mr-2" />
                      Chọn ảnh
                    </Button>
                  </div>
                </div>
              ) : (
                <div className="relative group">
                  <div className="aspect-square rounded-xl overflow-hidden bg-muted">
                    <img src={preview || "/placeholder.svg"} alt="Preview" className="w-full h-full object-cover" />
                  </div>
                  <button
                    onClick={clearImage}
                    className="absolute top-3 right-3 p-2 bg-card/90 backdrop-blur-sm rounded-full shadow-lg 
                             hover:bg-destructive hover:text-destructive-foreground transition-all"
                  >
                    <X className="w-4 h-4" />
                  </button>
                  <div className="absolute bottom-3 left-3 right-3">
                    <p
                      className="text-xs text-card bg-foreground/80 backdrop-blur-sm px-3 py-1.5 rounded-full 
                                truncate inline-block max-w-full"
                    >
                      {selectedImage?.name}
                    </p>
                  </div>
                </div>
              )}

              <input ref={fileInputRef} type="file" accept="image/*" onChange={handleImageSelect} className="hidden" />

              <Button
                onClick={handlePredict}
                disabled={!selectedImage || loading}
                className="w-full mt-6 h-12 bg-primary hover:bg-primary/90 text-primary-foreground font-medium"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                    Đang tạo mô tả...
                  </>
                ) : (
                  <>
                    <Sparkles className="w-5 h-5 mr-2" />
                    Tạo mô tả ngay
                  </>
                )}
              </Button>

              {error && (
                <div className="mt-4 p-4 bg-destructive/10 border border-destructive/20 rounded-lg">
                  <p className="text-sm text-destructive text-center">{error}</p>
                </div>
              )}
            </div>
          </div>

          {/* Right: Result Section */}
          <div className="bg-card rounded-2xl border border-border shadow-sm overflow-hidden lg:min-h-[500px]">
            <div className="p-5 md:p-6 border-b border-border">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-lg bg-accent flex items-center justify-center">
                  <MessageSquareText className="w-4 h-4 text-accent-foreground" />
                </div>
                <div>
                  <h2 className="font-semibold text-foreground">Mô tả ảnh</h2>
                  <p className="text-sm text-muted-foreground">Powered by AI</p>
                </div>
              </div>
            </div>

            <div className="p-5 md:p-6 min-h-[300px] flex items-center justify-center">
              {!result && !loading && (
                <div className="text-center py-8 md:py-12">
                  <div className="w-20 h-20 rounded-full bg-secondary mx-auto mb-4 flex items-center justify-center">
                    <MessageSquareText className="w-10 h-10 text-muted-foreground/50" />
                  </div>
                  <p className="text-muted-foreground">Mô tả ảnh sẽ hiển thị tại đây</p>
                  <p className="text-sm text-muted-foreground/70 mt-1">Tải lên ảnh và nhấn "Tạo mô tả ngay"</p>
                </div>
              )}

              {loading && (
                <div className="text-center py-8 md:py-12">
                  <div className="w-20 h-20 rounded-full bg-accent mx-auto mb-4 flex items-center justify-center animate-pulse">
                    <Loader2 className="w-10 h-10 text-primary animate-spin" />
                  </div>
                  <p className="text-foreground font-medium">Đang phân tích ảnh...</p>
                  <p className="text-sm text-muted-foreground mt-1">Vui lòng chờ trong giây lát</p>
                </div>
              )}

              {result && !loading && (
                <div className="w-full">
                  <div className="mb-6">
                    <div className="w-16 h-16 rounded-full bg-primary/10 mx-auto mb-4 flex items-center justify-center">
                      <MessageSquareText className="w-8 h-8 text-primary" />
                    </div>
                    <p className="text-sm text-muted-foreground uppercase tracking-wider mb-3 text-center">
                      Mô tả được tạo
                    </p>
                    <div className="bg-secondary/50 rounded-xl p-5 border border-border">
                      <p className="text-base md:text-lg text-foreground leading-relaxed">{result.caption}</p>
                    </div>
                  </div>

                  {result.confidence && (
                    <div className="bg-secondary rounded-xl p-4">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-sm text-muted-foreground">Độ tin cậy</span>
                        <span className="font-semibold text-primary">{(result.confidence * 100).toFixed(1)}%</span>
                      </div>
                      <div className="h-2 bg-muted rounded-full overflow-hidden">
                        <div
                          className="h-full bg-primary rounded-full transition-all duration-500"
                          style={{ width: `${result.confidence * 100}%` }}
                        />
                      </div>
                    </div>
                  )}

                  <Button variant="outline" className="w-full mt-4 bg-transparent" onClick={clearImage}>
                    Thử với ảnh khác
                  </Button>
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border mt-12 md:mt-16 py-8">
        <div className="max-w-7xl mx-auto px-6 text-center text-sm text-muted-foreground">
          <p>ImageCaption AI - Ứng dụng mô tả ảnh tự động bằng AI</p>
        </div>
      </footer>
    </main>
  )
}
