import yaml
import iup
import argparse

p = argparse.ArgumentParser()

p.add_argument("--config", type=str, default="config.yaml", dest="config")
args = p.parse_args()

args.config


parser = argparse.ArgumentParser(
    prog="ProgramName",
    description="What the program does",
    epilog="Text at the bottom of help",
)
