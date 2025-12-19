use crate::error::{BookmarksError, Result};
use std::env;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

pub struct OtelGuard {
    _dummy: (),
}

impl Drop for OtelGuard {
    fn drop(&mut self) {
        // cleanup if needed
    }
}

pub fn init_tracing(service_name: &str) -> Result<OtelGuard> {
    let enabled = env::var("BOOKMARKS_ENABLE_TRACING")
        .map(|v| {
            let v = v.to_lowercase();
            v == "1" || v == "true" || v == "yes"
        })
        .unwrap_or(false);

    let endpoint = env::var("PHOENIX_COLLECTOR_ENDPOINT")
        .or_else(|_| env::var("OTEL_EXPORTER_OTLP_ENDPOINT"))
        .ok();

    if !enabled && endpoint.is_none() {
        // tracing not enabled, just set up basic logging
        let subscriber = tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| "info".into()),
            )
            .finish();

        subscriber.init();

        tracing::info!("basic logging initialized (service={})", service_name);

        return Ok(OtelGuard { _dummy: () });
    }

    // for now, just use basic logging even if OTEL is enabled
    // full OTEL support would require additional configuration
    let subscriber = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .finish();

    subscriber.init();

    if let Some(ep) = endpoint {
        tracing::info!("tracing initialized for {} (would send to: {})", service_name, ep);
    } else {
        tracing::info!("tracing initialized for {}", service_name);
    }

    Ok(OtelGuard { _dummy: () })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_tracing_without_endpoint() {
        // should succeed but not enable otel
        let guard = init_tracing("test");
        assert!(guard.is_ok());
    }
}
