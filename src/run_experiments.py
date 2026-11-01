"""Run one paired source-style S=8 P&L experiment through the shared pipeline."""
from main import main as run_main


def main():
    run_main(experiment_name="paper_source_S8", base_seed=1, S=8)


if __name__ == "__main__":
    main()
