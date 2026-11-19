"""Build a people attribute table from Wikidata5M and Wikidata attributes."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml


GRAPH_FILES = [
    "wikidata5m_transductive_train.txt",
    "wikidata5m_transductive_valid.txt",
    "wikidata5m_transductive_test.txt",
]

HUMAN_TYPE = "Q5"
GENDER_PROPERTY = "P21"
CITIZENSHIP_PROPERTY = "P27"
OCCUPATION_PROPERTY = "P106"
AWARD_PROPERTY = "P166"


def read_humans(graph_dir: Path) -> set[str]:
    """Find entities that have (P31, Q5) in any graph split."""
    humans = set()

    for filename in GRAPH_FILES:
        path = graph_dir / filename
        print(f"Finding humans in {filename}...")

        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                parts = line.split()

                if len(parts) != 3:
                    raise ValueError(
                        f"Unexpected graph row in {path}, line {line_number}: "
                        f"{line.rstrip()}"
                    )

                subject, relation, obj = parts

                if relation == "P31" and obj == HUMAN_TYPE:
                    humans.add(subject)

    print(f"Distinct humans found: {len(humans):,}")
    return humans


def read_graph_properties(
    graph_dir: Path,
    humans: set[str],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Collect occupations and awards belonging to identified humans."""
    occupations = defaultdict(set)
    awards = defaultdict(set)

    for filename in GRAPH_FILES:
        path = graph_dir / filename
        print(f"Collecting occupations and awards from {filename}...")

        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                parts = line.split()

                if len(parts) != 3:
                    raise ValueError(
                        f"Unexpected graph row in {path}, line {line_number}"
                    )

                subject, relation, obj = parts

                if subject not in humans:
                    continue

                if relation == OCCUPATION_PROPERTY:
                    occupations[subject].add(obj)
                elif relation == AWARD_PROPERTY:
                    awards[subject].add(obj)

    return occupations, awards


def normalize_region_code(value: object) -> str | None:
    """Convert YAML region codes to three-character strings."""
    if value is None:
        return None

    code = str(value)
    if code.isdigit():
        return code.zfill(3)

    return code


def read_region_mapping(path: Path) -> tuple[dict[str, str], dict[str, dict]]:
    """Load approved country-to-region assignments from regions.yaml."""
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    entities = config["entities"]

    country_to_region = {}
    for country_qid, details in entities.items():
        region_code = normalize_region_code(details.get("region_code"))

        if (
            details.get("status") == "approved_working_decision"
            and region_code is not None
        ):
            country_to_region[country_qid] = region_code

    return country_to_region, entities


def read_attributes(
    path: Path,
    humans: set[str],
    country_to_region: dict[str, str],
) -> tuple[
    dict[str, set[str]],
    dict[str, set[str]],
    dict[str, set[str]],
]:
    """Read gender and citizenship attributes for identified humans."""
    genders = defaultdict(set)
    countries = defaultdict(set)
    regions = defaultdict(set)

    print("Reading gender and citizenship attributes...")

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")

        required_columns = {"qid", "property", "value"}
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Attributes file must contain columns: {required_columns}"
            )

        for row in reader:
            qid = row["qid"]
            property_id = row["property"]
            value = row["value"]

            if qid not in humans or not value:
                continue

            if property_id == GENDER_PROPERTY:
                genders[qid].add(value)

            elif property_id == CITIZENSHIP_PROPERTY:
                countries[qid].add(value)

                if value in country_to_region:
                    regions[qid].add(country_to_region[value])

    return genders, countries, regions


def write_audits(
    output_dir: Path,
    people: pd.DataFrame,
    region_config: dict[str, dict],
    country_to_region: dict[str, str],
    countries: dict[str, set[str]],
) -> None:
    """Write coverage, gender, region, and unmapped-citizenship audits."""
    total_humans = len(people)

    
    coverage = []
    for property_name, column in [
        ("P21", "gender"),
        ("P27", "countries"),
        ("P106", "occupations"),
        ("P166", "awards"),
    ]:
        if property_name == "P27":
            # Citizenship values are kept in the separate countries dictionary.
            count = sum(bool(countries.get(qid)) for qid in people["id"])
        else:
            count = int(people[column].map(bool).sum())

        coverage.append(
            {
                "property": property_name,
                "people_with_value": count,
                "people_without_value": total_humans - count,
                "coverage_percent": (
                    100 * count / total_humans if total_humans else 0
                ),
            }
        )

    pd.DataFrame(coverage).to_csv(
        output_dir / "people_coverage.csv", index=False
    )

    gender_counts = defaultdict(int)
    for values in people["gender"]:
        for value in values:
            gender_counts[value] += 1

    pd.DataFrame(
        [
            {"gender_qid": qid, "people_count": count}
            for qid, count in sorted(gender_counts.items())
        ]
    ).to_csv(output_dir / "people_gender_counts.csv", index=False)

    region_names = {
        normalize_region_code(code): details["name"]
        for code, details in yaml.safe_load(
            Path("configs/data/regions.yaml").read_text(encoding="utf-8")
        )["regions"].items()
    }

    region_counts = defaultdict(int)
    for values in people["regions"]:
        for value in values:
            region_counts[value] += 1

    pd.DataFrame(
        [
            {
                "region_code": code,
                "region_name": region_names.get(code, "Unknown"),
                "people_count": count,
            }
            for code, count in sorted(region_counts.items())
        ]
    ).to_csv(output_dir / "people_region_counts.csv", index=False)

    unmapped_counts = defaultdict(int)
    for person_countries in countries.values():
        for country in person_countries:
            if country not in country_to_region:
                unmapped_counts[country] += 1

    unmapped_rows = []
    for country, count in sorted(
        unmapped_counts.items(), key=lambda item: -item[1]
    ):
        details = region_config.get(country, {})
        unmapped_rows.append(
            {
                "country_qid": country,
                "label": details.get("label", ""),
                "status": details.get("status", "not_in_mapping"),
                "people_count": count,
            }
        )

    pd.DataFrame(unmapped_rows).to_csv(
        output_dir / "people_unmapped_citizenships.csv", index=False
    )

    print("\nAudit summary")
    print(f"Humans: {total_humans:,}")
    print(pd.DataFrame(coverage).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Wikidata5M people attribute table."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--regions-config",
        type=Path,
        default=Path("configs/data/regions.yaml"),
    )
    args = parser.parse_args()

    data_root = args.data_root
    graph_dir = data_root / "raw/wikidata5m"
    attributes_path = (
        data_root / "raw/wikidata_attributes/attributes_2026-10-25.tsv"
    )
    output_dir = data_root / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in [*[
        graph_dir / filename for filename in GRAPH_FILES
    ], attributes_path, args.regions_config]:
        if not path.exists():
            raise FileNotFoundError(f"Required input file not found: {path}")

    print("Step 1: Identifying humans...")
    humans = read_humans(graph_dir)

    print("\nStep 2: Collecting graph properties...")
    occupations, awards = read_graph_properties(graph_dir, humans)

    print("\nStep 3: Loading regional classification...")
    country_to_region, region_config = read_region_mapping(
        args.regions_config
    )

    print("\nStep 4: Reading attributes...")
    genders, countries, regions = read_attributes(
        attributes_path, humans, country_to_region
    )

    print("\nStep 5: Building people table...")
    people = pd.DataFrame(
        {
            "id": sorted(humans),
            "gender": [
                sorted(genders.get(qid, set())) for qid in sorted(humans)
            ],
            "regions": [
                sorted(regions.get(qid, set())) for qid in sorted(humans)
            ],
            "occupations": [
                sorted(occupations.get(qid, set())) for qid in sorted(humans)
            ],
            "awards": [
                sorted(awards.get(qid, set())) for qid in sorted(humans)
            ],
        }
    )

    output_path = output_dir / "people.parquet"
    people.to_parquet(output_path, index=False)
    print(f"Saved {len(people):,} people to {output_path}")

    print("\nStep 6: Writing audit tables...")
    write_audits(
        output_dir,
        people,
        region_config,
        country_to_region,
        countries,
    )

    print("\nFinished.")


if __name__ == "__main__":
    main()
