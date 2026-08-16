from .cli import main

# The guard is not optional here: training uses a process pool, and on macOS
# (spawn start method) every worker re-imports this module. Without it, each
# worker would launch its own training run.
if __name__ == "__main__":
    raise SystemExit(main())
