import argparse
import logging
from pathlib import Path



def run_vis_torch(args: argparse.Namespace) -> int:
    try:
        import torch
    except ImportError:
        logging.error("torch not installed; install torch to run this command")
        return 1

    artifact = args.artifact
    if not artifact.exists():
        logging.error("artifact %s does not exist", artifact)
        return 1

    try:
        payload = torch.load(artifact, map_location="cpu")
    except Exception as exc:  # pragma: no cover - runtime load errors
        logging.error("failed to load torch artifact: %s", exc)
        return 1

    embeddings = payload.get("embeddings")
    bookmarks = payload.get("bookmarks") or []

    if embeddings is None:
        logging.error("embeddings not found in artifact")
        return 1

    try:
        tensor = embeddings if isinstance(embeddings, torch.Tensor) else torch.tensor(embeddings)
    except Exception as exc:
        logging.error("embeddings could not be coerced to tensor: %s", exc)
        return 1

    count = tensor.shape[0] if tensor.ndim > 0 else 0
    dims = tensor.shape[1] if tensor.ndim > 1 else 0
    logging.info("loaded %s embeddings with dimension %s from %s", count, dims, artifact)

    show = min(args.limit, count)
    if show > 0 and bookmarks:
        logging.info("showing %s sample bookmark rows:", show)
        for idx in range(show):
            entry = bookmarks[idx] if idx < len(bookmarks) else {}
            logging.info("[%s] url=%s title=%s", idx, entry.get("url"), entry.get("title"))

    return 0


def register_vis_command(subparsers: argparse._SubParsersAction) -> None:
    vis_parser = subparsers.add_parser(
        "vis",
        help="visualize embeddings",
    )
    vis_subparsers = vis_parser.add_subparsers(dest="vis_command")

    summary_parser = vis_subparsers.add_parser(
        "summary",
        help="inspect a torch embedding artifact",
    )
    summary_parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("data/vectors.pt"),
        help="path to the torch artifact to inspect (default: data/vectors.pt)",
    )
    summary_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="how many sample bookmark rows to display",
    )
    summary_parser.set_defaults(command_handler=run_vis_torch)

    cluster_parser = vis_subparsers.add_parser(
        "cluster",
        help="cluster embeddings and print top tokens per cluster",
    )
    cluster_parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("data/vectors.pt"),
        help="path to the torch artifact to inspect (default: data/vectors.pt)",
    )
    cluster_parser.add_argument(
        "--clusters",
        type=int,
        default=6,
        help="how many clusters to derive (k-means)",
    )
    cluster_parser.add_argument(
        "--top-tokens",
        type=int,
        default=8,
        help="how many frequent tokens to show per cluster",
    )
    cluster_parser.add_argument(
        "--limit",
        type=int,
        default=2000,
        help="max points to cluster for speed",
    )
    cluster_parser.add_argument(
        "--include-stop-words",
        action="store_true",
        help="include stop words in token analysis",
    )
    cluster_parser.set_defaults(command_handler=run_vis_cluster)

    neighbors_parser = vis_subparsers.add_parser(
        "neighbors",
        help="show nearest neighbors for a bookmark (cosine similarity)",
    )
    neighbors_parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("data/vectors.pt"),
        help="path to the torch artifact to inspect (default: data/vectors.pt)",
    )
    neighbors_parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="row index to inspect (0-based)",
    )
    neighbors_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="how many nearest neighbors to show",
    )
    neighbors_parser.set_defaults(command_handler=run_vis_neighbors)

    organize_parser = vis_subparsers.add_parser(
        "organize",
        help="multi-resolution clustering for bookmark organization",
    )
    organize_parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("data/vectors.pt"),
        help="path to the torch artifact to inspect (default: data/vectors.pt)",
    )
    organize_parser.add_argument(
        "--resolutions",
        type=str,
        default="all,year,quarter",
        help="comma-separated resolutions: all,year,quarter,month (default: all,year,quarter)",
    )
    organize_parser.add_argument(
        "--clusters",
        type=int,
        default=6,
        help="how many clusters per resolution (default: 6)",
    )
    organize_parser.add_argument(
        "--top-tokens",
        type=int,
        default=8,
        help="how many frequent tokens to show per cluster",
    )
    organize_parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="how many sample bookmarks to include per cluster",
    )
    organize_parser.add_argument(
        "--output",
        type=Path,
        help="output file for organization results (JSON)",
    )
    organize_parser.set_defaults(command_handler=run_vis_organize)


def _load_torch_artifact(artifact: Path):
    try:
        import torch
    except ImportError:
        raise ImportError("torch not installed; install torch to run this command")

    if not artifact.exists():
        raise FileNotFoundError(f"artifact {artifact} does not exist")

    payload = torch.load(artifact, map_location="cpu")
    embeddings = payload.get("embeddings")
    bookmarks = payload.get("bookmarks") or []

    if embeddings is None:
        raise ValueError("embeddings not found in artifact")

    tensor = embeddings if isinstance(embeddings, torch.Tensor) else torch.tensor(embeddings)
    return tensor, bookmarks


def _tokenize(text: str, include_stop_words: bool = False) -> list[str]:
    import string

    tokens = [chunk for chunk in text.lower().split() if chunk.isascii()]

    # Filter pure punctuation tokens (common title separators)
    tokens = [tok for tok in tokens if not all(c in string.punctuation for c in tok)]

    if include_stop_words:
        return tokens

    import nltk
    try:
        stop_set = set(nltk.corpus.stopwords.words("english"))
    except LookupError:
        # Download stopwords if not available
        nltk.download("stopwords", quiet=True)
        stop_set = set(nltk.corpus.stopwords.words("english"))

    return [tok for tok in tokens if tok not in stop_set]


def _kmeans(tensor, clusters: int, iterations: int = 10) -> list[int]:
    import torch

    count = tensor.shape[0]
    k = min(max(clusters, 1), count)
    # init centroids from first k points for determinism
    try:
        centroids = tensor[:k].clone()
    except TypeError:
        centroids = torch.tensor(list(tensor)[:k]).clone()

    for _ in range(iterations):
        # assign
        distances = torch.cdist(tensor, centroids, p=2)
        labels = distances.argmin(dim=1)
        # update
        try:
            for idx in range(k):
                mask = labels == idx
                if mask.any():
                    centroids[idx] = tensor[mask].mean(dim=0)
        except Exception:
            label_rows = list(labels)
            label_values = [
                int(row[0]) if isinstance(row, (list, tuple)) and row else int(row)
                for row in label_rows
            ]
            points = list(tensor)
            for idx in range(k):
                assigned = [row for row, label in zip(points, label_values, strict=False) if label == idx]
                if not assigned:
                    continue
                cols = list(zip(*assigned))
                centroids.data[idx] = [sum(col) / len(col) for col in cols]
    if hasattr(labels, "tolist"):
        return labels.tolist()
    label_rows = list(labels)
    return [int(row[0]) if isinstance(row, (list, tuple)) and row else int(row) for row in label_rows]


def _format_timestamp(ts_microseconds: int | None) -> str:
    """Convert microsecond timestamp to human-readable date."""
    if ts_microseconds is None:
        return "unknown"
    try:
        from datetime import datetime
        # Firefox timestamps are in microseconds
        ts_seconds = ts_microseconds / 1_000_000 if ts_microseconds > 10_000_000_000 else ts_microseconds
        dt = datetime.fromtimestamp(ts_seconds)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return "invalid"


def run_vis_cluster(args: argparse.Namespace) -> int:
    try:
        import torch
    except ImportError:
        logging.error("torch not installed; install torch to run this command")
        return 1

    try:
        tensor, bookmarks = _load_torch_artifact(args.artifact)
    except (ImportError, FileNotFoundError, ValueError) as exc:
        logging.error("%s", exc)
        return 1
    except Exception as exc:  # pragma: no cover
        logging.error("failed to load artifact: %s", exc)
        return 1

    count = tensor.shape[0]
    if count == 0:
        logging.error("no embeddings to cluster")
        return 1

    capped = min(args.limit, count)
    if capped < count:
        logging.info("limiting to first %s points for clustering", capped)
        tensor = tensor[:capped]
        bookmarks = bookmarks[:capped]

    labels = _kmeans(tensor, clusters=args.clusters)
    include_stop_words = getattr(args, "include_stop_words", False)
    buckets: dict[int, list[tuple[str, int | None]]] = {}
    for idx, label in enumerate(labels):
        title = bookmarks[idx].get("title") or bookmarks[idx].get("url") or ""
        timestamp = bookmarks[idx].get("timestamp")
        buckets.setdefault(label, []).append((title, timestamp))

    for label, items in sorted(buckets.items()):
        all_tokens: list[str] = []
        timestamps: list[int] = []
        for title, ts in items:
            all_tokens.extend(_tokenize(title, include_stop_words=include_stop_words))
            if ts is not None:
                timestamps.append(ts)

        counter: dict[str, int] = {}
        for tok in all_tokens:
            counter[tok] = counter.get(tok, 0) + 1
        top_tokens = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)[: args.top_tokens]
        token_text = ", ".join(tok for tok, _ in top_tokens) or "(no tokens)"

        # Calculate temporal range
        if timestamps:
            min_ts = min(timestamps)
            max_ts = max(timestamps)
            date_range = f"{_format_timestamp(min_ts)} to {_format_timestamp(max_ts)}"
        else:
            date_range = "no dates"

        logging.info("cluster %s (%s items, %s): %s", label, len(items), date_range, token_text)

    return 0


def _cosine_similarity(tensor, index: int):
    import torch

    vector = tensor[index]
    normed = torch.nn.functional.normalize(tensor, dim=1)
    sims = torch.matmul(normed, normed[index])
    if hasattr(sims, "tolist"):
        return sims.tolist()
    return list(sims)


def _partition_by_time(bookmarks: list[dict], resolution: str) -> dict[str, list[tuple[int, dict]]]:
    """Partition bookmarks by time resolution, returning {period: [(index, bookmark), ...]}"""
    from datetime import datetime

    partitions: dict[str, list[tuple[int, dict]]] = {}

    for idx, bookmark in enumerate(bookmarks):
        ts = bookmark.get("timestamp")
        if ts is None:
            # Put bookmarks without timestamps in 'unknown' partition
            partitions.setdefault("unknown", []).append((idx, bookmark))
            continue

        # Convert microseconds to datetime
        ts_seconds = ts / 1_000_000 if ts > 10_000_000_000 else ts
        try:
            dt = datetime.fromtimestamp(ts_seconds)
        except (ValueError, OSError):
            partitions.setdefault("unknown", []).append((idx, bookmark))
            continue

        # Determine partition key based on resolution
        if resolution == "year":
            key = str(dt.year)
        elif resolution == "quarter":
            quarter = (dt.month - 1) // 3 + 1
            key = f"{dt.year}-Q{quarter}"
        elif resolution == "month":
            key = f"{dt.year}-{dt.month:02d}"
        else:  # "all" or unknown
            key = "all"

        partitions.setdefault(key, []).append((idx, bookmark))

    return partitions


def run_vis_organize(args: argparse.Namespace) -> int:
    import json

    try:
        import torch
    except ImportError:
        logging.error("torch not installed; install torch to run this command")
        return 1

    try:
        tensor, bookmarks = _load_torch_artifact(args.artifact)
    except (ImportError, FileNotFoundError, ValueError) as exc:
        logging.error("%s", exc)
        return 1
    except Exception as exc:  # pragma: no cover
        logging.error("failed to load artifact: %s", exc)
        return 1

    count = tensor.shape[0]
    if count == 0:
        logging.error("no embeddings available")
        return 1

    resolutions = [r.strip() for r in args.resolutions.split(",")]
    results: dict[str, dict] = {}

    for resolution in resolutions:
        logging.info("processing resolution: %s", resolution)

        if resolution == "all":
            # Single clustering of all bookmarks
            partitions = {"all": [(i, bookmarks[i]) for i in range(len(bookmarks))]}
        else:
            # Partition by time
            partitions = _partition_by_time(bookmarks, resolution)

        resolution_results: dict[str, dict] = {}

        for partition_key, items in sorted(partitions.items()):
            if len(items) < args.clusters:
                logging.warning("partition %s has only %s items, skipping", partition_key, len(items))
                continue

            # Extract indices and bookmarks
            indices = [idx for idx, _ in items]
            partition_bookmarks = [bm for _, bm in items]

            # Cluster this partition
            partition_tensor = tensor[indices]
            labels = _kmeans(partition_tensor, clusters=args.clusters)

            # Analyze clusters
            buckets: dict[int, list[tuple[int, str, int | None]]] = {}
            for local_idx, label in enumerate(labels):
                global_idx = indices[local_idx]
                bookmark = partition_bookmarks[local_idx]
                title = bookmark.get("title") or bookmark.get("url") or ""
                timestamp = bookmark.get("timestamp")
                buckets.setdefault(label, []).append((global_idx, title, timestamp))

            clusters_info = []
            for cluster_id, cluster_items in sorted(buckets.items()):
                # Token analysis
                all_tokens: list[str] = []
                timestamps: list[int] = []
                for _, title, ts in cluster_items:
                    all_tokens.extend(_tokenize(title, include_stop_words=False))
                    if ts is not None:
                        timestamps.append(ts)

                counter: dict[str, int] = {}
                for tok in all_tokens:
                    counter[tok] = counter.get(tok, 0) + 1
                top_tokens = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)[: args.top_tokens]

                # Date range
                if timestamps:
                    min_ts = min(timestamps)
                    max_ts = max(timestamps)
                    date_range = f"{_format_timestamp(min_ts)} to {_format_timestamp(max_ts)}"
                else:
                    date_range = "no dates"

                # Sample bookmarks
                samples = []
                for idx, title, ts in cluster_items[: args.samples]:
                    samples.append({
                        "index": idx,
                        "title": title,
                        "date": _format_timestamp(ts),
                        "url": bookmarks[idx].get("url"),
                    })

                cluster_info = {
                    "cluster_id": cluster_id,
                    "count": len(cluster_items),
                    "date_range": date_range,
                    "top_tokens": [tok for tok, _ in top_tokens],
                    "samples": samples,
                    "bookmark_indices": [idx for idx, _, _ in cluster_items],
                }
                clusters_info.append(cluster_info)

                # Log summary
                token_text = ", ".join(tok for tok, _ in top_tokens) or "(no tokens)"
                logging.info("  %s cluster %s (%s items, %s): %s", partition_key, cluster_id, len(cluster_items), date_range, token_text)

            resolution_results[partition_key] = {
                "count": len(items),
                "clusters": clusters_info,
            }

        results[resolution] = resolution_results

    # Output results
    output_data = {
        "resolutions": results,
        "total_bookmarks": count,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(output_data, f, indent=2)
        logging.info("wrote multi-resolution clustering to %s", args.output)
    else:
        # Print to stdout
        print(json.dumps(output_data, indent=2))

    return 0


def run_vis_neighbors(args: argparse.Namespace) -> int:
    try:
        import torch
    except ImportError:
        logging.error("torch not installed; install torch to run this command")
        return 1

    try:
        tensor, bookmarks = _load_torch_artifact(args.artifact)
    except (ImportError, FileNotFoundError, ValueError) as exc:
        logging.error("%s", exc)
        return 1
    except Exception as exc:  # pragma: no cover
        logging.error("failed to load artifact: %s", exc)
        return 1

    count = tensor.shape[0]
    if count == 0:
        logging.error("no embeddings available")
        return 1

    target_idx = args.index
    if target_idx < 0 or target_idx >= count:
        logging.error("target index %s out of range (0-%s)", target_idx, count - 1)
        return 1

    sims = _cosine_similarity(tensor, target_idx)
    ranked = sorted(
        [(idx, score) for idx, score in enumerate(sims) if idx != target_idx],
        key=lambda pair: pair[1],
        reverse=True,
    )[: args.top_k]

    target = bookmarks[target_idx]
    target_date = _format_timestamp(target.get("timestamp"))
    logging.info("neighbors for [%s] (%s) url=%s title=%s", target_idx, target_date, target.get("url"), target.get("title"))
    for idx, score in ranked:
        entry = bookmarks[idx]
        entry_date = _format_timestamp(entry.get("timestamp"))
        logging.info("  -> [%s] score=%.4f (%s) url=%s title=%s", idx, score, entry_date, entry.get("url"), entry.get("title"))

    return 0
