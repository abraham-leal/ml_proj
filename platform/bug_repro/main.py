#!/usr/bin/env python
import wandb
import random
import sys

wandb.require("core")
def main():

    with wandb.init(entity="meta-repro", project="meta-bug-repro", id="stress_test_run-5-"+str(sys.stdin.readlines()[0]), resume="allow") as run:
        run.define_metric("*", step_metric="global_step")
        offset = random.random() / 5
        for epoch in range(1, 1000001):
            values = {}
            values["global_step"] = epoch
            for metric_ind in range(1, 3000):
                values[f"acc_{metric_ind}"] = 1 - 2 ** -epoch - random.random() / epoch - offset
                values[f"loss_{metric_ind}"] = 2 ** -epoch + random.random() / epoch + offset
            print("logging " + str(epoch))
            run.log(values)

if __name__ == '__main__':
    main()