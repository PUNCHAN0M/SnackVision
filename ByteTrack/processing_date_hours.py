"""
CSV Data Aggregation Tool

This script processes CSV files from date-based folders in the results directory
and creates a summary CSV with aggregated counts by date.

Input structure: ./results/{date}/csv/*.csv
Output: ./csvdata/summary_data.csv

Expected CSV format:
Input:  tracker_id,count_food,count_snack,count_drink,count_glass,count_person
Output: date,drink,snack,food,glass
"""

import os
import csv
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CSVAggregator:
    """Aggregates CSV data from date-based folder structure."""

    def __init__(
        self,
        results_path: str = "./results_test_after",
        output_path: str = "./csvdata/summary_data.csv",
    ):
        self.results_path = Path(results_path)
        self.output_path = Path(output_path)

        # Create output directory if it doesn't exist
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def parse_date_folder(self, folder_name: str) -> str:
        """
        Convert folder name from YYYYMMDD to YYYY-MM-DD format.

        Args:
            folder_name: Folder name in format YYYYMMDD (e.g., "20250915")

        Returns:
            Formatted date string YYYY-MM-DD (e.g., "2025-09-15")
        """
        try:
            # Parse YYYYMMDD format
            date_obj = datetime.strptime(folder_name, "%Y%m%d")
            return date_obj.strftime("%Y-%m-%d")
        except ValueError:
            logger.warning(f"Could not parse date from folder name: {folder_name}")
            return folder_name

    def read_csv_file(self, csv_file_path: Path) -> Dict[str, int]:
        """
        Read a single CSV file and aggregate the counts.

        Args:
            csv_file_path: Path to the CSV file

        Returns:
            Dictionary with aggregated counts for each category
        """
        counts = {"food": 0, "snack": 0, "drink": 0, "glass": 0}

        try:
            with open(csv_file_path, "r", encoding="utf-8") as file:
                reader = csv.DictReader(file)

                for row in reader:
                    # Skip header row or empty rows
                    if not row or "tracker_id" not in row:
                        continue

                    # Aggregate counts for each category
                    counts["food"] += int(row.get("count_food", 0))
                    counts["snack"] += int(row.get("count_snack", 0))
                    counts["drink"] += int(row.get("count_drink", 0))
                    counts["glass"] += int(row.get("count_glass", 0))

        except Exception as e:
            logger.error(f"Error reading {csv_file_path}: {e}")

        return counts

    def process_date_folder(self, date_folder_path: Path) -> Tuple[str, Dict[str, int]]:
        """
        Process all CSV files in a date folder and aggregate counts.

        Args:
            date_folder_path: Path to the date folder

        Returns:
            Tuple of (formatted_date, aggregated_counts)
        """
        folder_name = date_folder_path.name
        formatted_date = self.parse_date_folder(folder_name)

        # Initialize total counts for this date
        total_counts = {"food": 0, "snack": 0, "drink": 0, "glass": 0}

        # Look for CSV files in the csv subfolder
        csv_folder = date_folder_path / "csv"
        if not csv_folder.exists():
            logger.warning(f"CSV folder not found: {csv_folder}")
            return formatted_date, total_counts

        csv_files = list(csv_folder.glob("*.csv"))
        logger.info(f"Processing {len(csv_files)} CSV files in {date_folder_path}")

        for csv_file in csv_files:
            file_counts = self.read_csv_file(csv_file)

            # Add to total counts
            for category in total_counts:
                total_counts[category] += file_counts[category]

        logger.info(f"Date {formatted_date}: {total_counts}")
        return formatted_date, total_counts

    def aggregate_all_data(self) -> List[Dict[str, any]]:
        """
        Process all date folders and aggregate data.

        Returns:
            List of dictionaries with aggregated data by date
        """
        if not self.results_path.exists():
            logger.error(f"Results path does not exist: {self.results_path}")
            return []

        summary_data = []

        # Get all date folders (folders with YYYYMMDD pattern)
        date_folders = [
            folder
            for folder in self.results_path.iterdir()
            if folder.is_dir() and folder.name.isdigit() and len(folder.name) == 8
        ]

        if not date_folders:
            logger.warning(f"No date folders found in {self.results_path}")
            return []

        # Sort folders by date
        date_folders.sort(key=lambda x: x.name)

        for date_folder in date_folders:
            formatted_date, counts = self.process_date_folder(date_folder)

            summary_data.append(
                {
                    "date": formatted_date,
                    "drink": counts["drink"],
                    "snack": counts["snack"],
                    "food": counts["food"],
                    "glass": counts["glass"],
                }
            )

        return summary_data

    def write_summary_csv(self, summary_data: List[Dict[str, any]]) -> None:
        """
        Write the aggregated data to the output CSV file.

        Args:
            summary_data: List of dictionaries with aggregated data
        """
        if not summary_data:
            logger.warning("No data to write to CSV")
            return

        try:
            with open(self.output_path, "w", newline="", encoding="utf-8") as file:
                fieldnames = ["date", "drink", "snack", "food", "glass"]
                writer = csv.DictWriter(file, fieldnames=fieldnames)

                writer.writeheader()
                writer.writerows(summary_data)

            logger.info(f"Summary CSV written to: {self.output_path}")
            logger.info(f"Total records: {len(summary_data)}")

        except Exception as e:
            logger.error(f"Error writing summary CSV: {e}")

    def run(self) -> None:
        """Run the complete aggregation process."""
        logger.info("Starting CSV data aggregation...")
        logger.info(f"Results path: {self.results_path}")
        logger.info(f"Output path: {self.output_path}")

        # Aggregate all data
        summary_data = self.aggregate_all_data()

        if summary_data:
            # Write summary CSV
            self.write_summary_csv(summary_data)

            # Print summary statistics
            total_drink = sum(row["drink"] for row in summary_data)
            total_snack = sum(row["snack"] for row in summary_data)
            total_food = sum(row["food"] for row in summary_data)
            total_glass = sum(row["glass"] for row in summary_data)

            logger.info("\n" + "=" * 50)
            logger.info("AGGREGATION SUMMARY")
            logger.info("=" * 50)
            logger.info(f"Dates processed: {len(summary_data)}")
            logger.info(f"Total drink count: {total_drink}")
            logger.info(f"Total snack count: {total_snack}")
            logger.info(f"Total food count: {total_food}")
            logger.info(f"Total glass count: {total_glass}")
            logger.info("=" * 50)
        else:
            logger.error(
                "No data was processed. Please check the folder structure and CSV files."
            )


def preview_folder_structure(results_path: str = "./results") -> None:
    """Preview the folder structure to help debug path issues."""
    results_path = Path(results_path)

    print(f"\n{'='*60}")
    print(f"FOLDER STRUCTURE PREVIEW: {results_path.absolute()}")
    print("=" * 60)

    if not results_path.exists():
        print(f"❌ Results path does not exist: {results_path}")
        return

    date_folders = []
    other_folders = []

    for item in results_path.iterdir():
        if item.is_dir():
            if item.name.isdigit() and len(item.name) == 8:
                date_folders.append(item)
            else:
                other_folders.append(item)

    print(f"\n📂 Date folders found ({len(date_folders)}):")
    for folder in sorted(date_folders, key=lambda x: x.name):
        csv_folder = folder / "csv"
        csv_count = len(list(csv_folder.glob("*.csv"))) if csv_folder.exists() else 0
        formatted_date = datetime.strptime(folder.name, "%Y%m%d").strftime("%Y-%m-%d")
        print(f"  📅 {folder.name} → {formatted_date} ({csv_count} CSV files)")

    if other_folders:
        print(f"\n📂 Other folders ({len(other_folders)}):")
        for folder in other_folders:
            print(f"  📁 {folder.name}")

    print("=" * 60)


def main():
    """Main function to run the CSV aggregation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Aggregate CSV data from date-based folders"
    )
    parser.add_argument(
        "--results-path",
        default="./results",
        help="Path to results folder (default: ./results)",
    )
    parser.add_argument(
        "--output-path",
        default="./csvdata/summary_data.csv",
        help="Output CSV path (default: ./csvdata/summary_data.csv)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview folder structure without processing",
    )

    args = parser.parse_args()

    if args.preview:
        preview_folder_structure(args.results_path)
        return

    aggregator = CSVAggregator(args.results_path, args.output_path)
    aggregator.run()


if __name__ == "__main__":
    main()
