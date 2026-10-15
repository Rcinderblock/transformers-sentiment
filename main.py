"""Запуск этапов проекта из одной команды."""

import argparse
import logging

import torch


def main():
    parser = argparse.ArgumentParser(description="Transformer sentiment analysis")
    parser.add_argument(
        "--day", choices=[str(i) for i in range(1, 8)] + ["all"], required=True
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=2000,
        help="Размер основного набора IMDB; test — 20%, validation берётся отдельно",
    )
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    torch.set_num_threads(4)
    from comparison import analyze_errors, compare
    from data import prepare_data
    from exploration import day1, day2, day3
    from training import fine_tune, train_baseline

    stages = {
        "1": day1,
        "2": lambda: day2(args.device),
        "3": lambda: day3(args.device),
        "4": lambda: (prepare_data(args.size), train_baseline(args.device)),
        "5": lambda: fine_tune(args.device, args.epochs),
        "6": lambda: compare(args.device),
        "7": analyze_errors,
    }
    try:
        for day in stages if args.day == "all" else [args.day]:
            print(f"=== Day {day} ===", flush=True)
            stages[day]()
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Ошибка: {exc}\n")


if __name__ == "__main__":
    main()
