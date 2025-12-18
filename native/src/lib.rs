use pyo3::prelude::*;

#[pyfunction]
fn normalize_firefox_timestamp(timestamp: Option<i64>) -> Option<i64> {
    let ts = timestamp?;
    if ts > 10_000_000_000 {
        Some(ts / 1_000_000)
    } else {
        Some(ts)
    }
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(normalize_firefox_timestamp, module)?)?;
    Ok(())
}

