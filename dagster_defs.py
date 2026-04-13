# Basic Imports
import datetime as dt
import os
from pathlib import Path

# Direct use of dagster
import dagster as dg
import requests
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from cfa_dagster import (
    ADLS2PickleIOManager,
    DynamicGraphAssetExecutionContext,
    ExecutionConfig,
    SelectorConfig,
    azure_batch_executor,
    azure_container_app_job_executor,
    collect_definitions,
    docker_executor,
    dynamic_executor,
    dynamic_graph_asset,
    start_dev_env,
)
from dagster_azure.blob import (
    AzureBlobStorageDefaultCredential,
    AzureBlobStorageResource,
)

# ============================================================================
# DAGSTER INITIALIZATION
# ============================================================================

# function to start the dev server
start_dev_env(__name__)

# env variable set by Dagster CLI
is_production: bool = not os.getenv("DAGSTER_IS_DEV_CLI")

# get the user running the Dagster instance
user = os.getenv("DAGSTER_USER")

# ============================================================================
# RUNTIME CONFIGURATION: WORKING DIRECTORY, EXECUTORS
# ============================================================================
# Executors define the runtime-location of an asset job
# See later on for Asset job definitions

# ---------- Working Directory, Branch, and Image Tag ----------

workdir = "cfa-immunization-uptake-projection"
local_workdir = Path(__file__).parent.resolve()

# If the tag is prod, use 'latest'.
# Else iteratively test on our dev images
# (You can always manually specify an override in the GUI)
try:
    print("You are running inside a .git repository; getting branchname from .git")
    repo = Repository(os.getcwd())
    current_branch_name = str(repo.head.shorthand)
    git_commit_sha = str(repo.head.target)
except Exception:
    print(
        "No .git folder detected; attempting to get branch name from build-arg $GIT_BRANCH_NAME"
    )
    current_branch_name = os.getenv("GIT_BRANCH_NAME", "unknown_branch")
    git_commit_sha = os.getenv("GIT_COMMIT_SHA", "unknown_commit_hash")

print(f"Current branch name is {current_branch_name}")

tag = (
    "latest"
    if (is_production or current_branch_name == "main")
    else current_branch_name
)
image = f"ghcr.io/cdcgov/cfa-immunization-uptake-projection:{tag}"

# ---------- Execution Configuration ----------

# Most basic execution - in dev, launches and runs locally
# In prod, launches on the code location but runs in Azure Container App Jobs
# Used for lightweight assets and jobs, etc. where volume mounts are not needed
basic_execution_config = ExecutionConfig(
    executor=SelectorConfig(
        class_name=azure_container_app_job_executor.__name__
        if is_production
        else dg.multiprocess_executor.__name__
    ),
)

# Launches locally, executes in a docker container as configured below
# Allows for rapid local testing in a similar-to-batch environment
docker_execution_config = ExecutionConfig(
    executor=SelectorConfig(
        class_name=docker_executor.__name__,
        config={
            "image": image,
            "env_vars": [
                f"DAGSTER_USER={user}",
                "VIRTUAL_ENV=/cfa-immunization-uptake-projection/.venv",
            ],
            "retries": {"enabled": {}},
            "container_kwargs": {
                "volumes": [
                    # bind the ~/.azure folder for optional cli login
                    f"/home/{user}/.azure:/root/.azure",
                    # bind current file so we don't have to rebuild
                    # the container image for workflow changes
                    f"{__file__}:/{workdir}/{os.path.basename(__file__)}",
                ]
            },
        },
    ),
)

# ============================================================================
# ASSET DEFINITIONS
# ============================================================================
# These are the core of Dagster - functions that specify data

# Daily Partitions
daily_partitions_def = dg.DailyPartitionsDefinition(
    start_date="2026-01-01", end_offset=1, timezone="America/New_York"
)

@dg.asset(partitions_def=daily_partitions_def)
def preprocess(context: dg.AssetExecutionContext):
    return

@dg.asset(partitions_def=daily_partitions_def)
def plot_data(context: dg.AssetExecutionContext, preprocess):
    return

@dg.asset(partitions_def=daily_partitions_def)
def fit(context: dg.AssetExecutionContext, plot_data):
    return

@dg.asset(partitions_def=daily_partitions_def)
def predict(context: dg.AssetExecutionContext, fit):
    return

@dg.asset(partitions_def=daily_partitions_def)
def plot_preds(context: dg.AssetExecutionContext, predict):
    return

@dg.asset(partitions_def=daily_partitions_def)
def eval(context: dg.AssetExecutionContext, plot_preds):
    return


@dg.asset(partitions_def=daily_partitions_def)
def plot_scores(context: dg.AssetExecutionContext, eval):
    return


# ============================================================================
# DAGSTER DEFINITIONS OBJECT
# ============================================================================
# This code allows us to collect all of the above definitions into a single
# Definitions object for Dagster to read. By doing this, we can keep our
# Dagster code in a single file instead of splitting it across multiple files.

# change storage accounts between dev and prod
storage_account = "cfadagster" if is_production else "cfadagsterdev"

# collect Dagster definitions from the current file
collected_defs = collect_definitions(globals())


# Create Definitions object
defs = dg.Definitions(
    assets=collected_defs["assets"],
    asset_checks=collected_defs["asset_checks"],
    jobs=collected_defs["jobs"],
    sensors=collected_defs["sensors"],
    schedules=collected_defs["schedules"],
    resources={
        # This IOManager lets Dagster serialize asset outputs and store them
        # in Azure to pass between assets
        "io_manager": ADLS2PickleIOManager(),
        # an example storage account
        "azure_blob_storage": AzureBlobStorageResource(
            account_url=f"{storage_account}.blob.core.windows.net",
            credential=AzureBlobStorageDefaultCredential(),
        ),
    },
    # You can put a comment after azure_batch_config to solely execute with Azure batch
    executor=dynamic_executor(
        default_config=basic_execution_config,
        alternate_configs=[docker_execution_config]
    ),
)
