"use client";

import { useState } from "react";

export default function Home() {
  const [leftImage, setLeftImage] = useState<File | null>(null);
  const [rightImage, setRightImage] = useState<File | null>(null);
  const [leftPreview, setLeftPreview] = useState<string | null>(null);
  const [rightPreview, setRightPreview] = useState<string | null>(null);
  const [result, setResult] = useState<string>("");
  const [loading, setLoading] = useState(false);

  const handleFileChange = (
    file: File | null,
    side: "left" | "right"
  ) => {
    if (!file) return;

    const previewUrl = URL.createObjectURL(file);

    if (side === "left") {
      setLeftImage(file);
      setLeftPreview(previewUrl);
    } else {
      setRightImage(file);
      setRightPreview(previewUrl);
    }
  };

  const handleSubmit = async () => {
    if (!leftImage || !rightImage) {
      setResult("Please upload both left and right eye images.");
      return;
    }

    setLoading(true);
    setResult("");

    const formData = new FormData();
    formData.append("leftImage", leftImage);
    formData.append("rightImage", rightImage);

    try {
      const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        setResult("Error: " + JSON.stringify(data));
        return;
      }

      setResult(data.result);
    } catch (error) {
      setResult("Could not connect to backend. Make sure FastAPI is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-gray-50 flex items-center justify-center p-6">
      <div className="w-full max-w-4xl bg-white rounded-2xl shadow-lg p-8">
        <h1 className="text-3xl font-bold text-center mb-2">
          脳損傷検出器
        </h1>

        <p className="text-center text-gray-600 mb-8">
        左右の目の画像をアップロードして瞳孔の違いを確認します
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="border rounded-xl p-4">
            <h2 className="font-semibold mb-3">左目の画像</h2>

            <input
              type="file"
              accept="image/*"
              onChange={(e) =>
                handleFileChange(e.target.files?.[0] || null, "left")
              }
            />

            {leftPreview && (
              <img
                src={leftPreview}
                alt="Left eye preview"
                className="mt-4 w-full h-64 object-contain rounded-lg border"
              />
            )}
          </div>

          <div className="border rounded-xl p-4">
            <h2 className="font-semibold mb-3">右目の画像</h2>

            <input
              type="file"
              accept="image/*"
              onChange={(e) =>
                handleFileChange(e.target.files?.[0] || null, "right")
              }
            />

            {rightPreview && (
              <img
                src={rightPreview}
                alt="Right eye preview"
                className="mt-4 w-full h-64 object-contain rounded-lg border"
              />
            )}
          </div>
        </div>

        <button
          onClick={handleSubmit}
          disabled={loading}
          className="mt-8 w-full bg-black text-white py-3 rounded-xl font-semibold hover:bg-gray-800 disabled:bg-gray-400"
        >
          {loading ? "Analyzing..." : "Analyze Images"}
        </button>

        {result && (
          <div className="mt-6 p-5 rounded-xl bg-gray-100 text-center font-medium">
            {result}
          </div>
        )}
      </div>
    </main>
  );
}