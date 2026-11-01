//! Error type shared by the request handlers.

use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};

/// Wraps any error into a `400 Bad Request` JSON body `{ "error": "…" }`.
pub struct AppError(pub anyhow::Error);

impl<E: Into<anyhow::Error>> From<E> for AppError {
    fn from(e: E) -> Self {
        Self(e.into())
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let body = serde_json::json!({ "error": self.0.to_string() });
        (StatusCode::BAD_REQUEST, Json(body)).into_response()
    }
}

pub type AppResult<T> = Result<T, AppError>;
