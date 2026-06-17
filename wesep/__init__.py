def __getattr__(name):
    if name in ("load_model", "load_model_local"):
        from wesep.cli.extractor import load_model, load_model_local

        return {
            "load_model": load_model,
            "load_model_local": load_model_local,
        }[name]
    raise AttributeError(f"module 'wesep' has no attribute {name!r}")


__all__ = ["load_model", "load_model_local"]
