import wandb


def _find_last_logged(args, logger):
    api = wandb.Api()
    try:
        run = api.run(f"{args.wandb_team}/{args.project}/{args.run_id}")
        logger.info(f"The last step recorded was: {run.lastHistoryStep}")
        logger.info(run.summary)
        logger.info(run.lastHistoryStep)
        if run.lastHistoryStep is None:
            logger.info("\nNo data was uploaded to WandB for this run before")
            return {"timestamp": 0}
        # Run.summary might not always have a in case no new data got uploaded to WandB
        # Hence we need to query the history
        history = run.scan_history(keys=["timestamp"], min_step=run.lastHistoryStep)
        _last_step = None
        highest_timestamp = 0
        for _last_step in history:
            if _last_step["timestamp"] > highest_timestamp:
                logger.info(f"\nLast step: {_last_step}")
                highest_timestamp = _last_step["timestamp"]
        if _last_step is None:
            logger.info("No data was uploaded to WandB for this run before")
            return {"timestamp": 0}
        if run.summary["timestamp"] > highest_timestamp:
            logger.info("Overiding the last step with the run.summary timestamp")
            highest_timestamp = run.summary["timestamp"]
        return {"timestamp": highest_timestamp}
    except Exception as e:
        logger.error(
            f"\nException while querying last run: {e}. \nWill upload all data."
        )
        return {"timestamp": 0}


def main():

    overrides = {
        "base_url": "http://k8s-default-appingre-6226b38256-166782791.us-east-1.elb.amazonaws.com",
        "entity": "alealwandb",
        "project": "meta-bug-repro"
    }

    api = wandb.Api(api_key="local-017febd67d206c09d4bda0f66d949f02412e7a92", overrides=overrides, timeout=60)

    run = api.run("stress_test_run-4-4")
    history = run.scan_history(page_size=6000, max_step=50)
    losses = [row["loss_200"] for row in history]

    print(history)



if __name__ == '__main__':
    main()