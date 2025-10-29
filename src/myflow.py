from src.logger import setup_logger
from src.args import args
import mlflow

from mlflow.exceptions import MlflowTracingException

logger = setup_logger()
_nolog = False


def connect(
    run_name=None,
    uri="http://localhost:5000",
    experiment_name="Ansatz Experiment",
    log_system_metrics=False,
    nolog=False,
):
    if nolog:
        global _nolog
        _nolog = True

        return

    mlflow.set_tracking_uri(uri=args.uri)
    mlflow.set_experiment(experiment_name)
    mlflow.start_run(
        run_name=run_name,
        log_system_metrics=log_system_metrics,
    )


def end_run():
    if _nolog:
        return

    mlflow.end_run()


def resume_run(mlflowrun):
    if _nolog:
        return

    run_id = mlflowrun.info.run_id
    mlflow.start_run(run_id=run_id, log_system_metrics=True)


def active_run():
    if _nolog:
        return

    return mlflow.active_run()


def start_nested(name):
    if _nolog:
        return

    try:
        mlflow.start_run(run_name=name, log_system_metrics=True, nested=True)

    except MlflowTracingException as e:
        logger.error("CANNOT START NESTED RUN")
        logger.debug("CANNOT START NESTED RUN: %s", e)


def end_nested():
    if _nolog:
        return

    mlflow.end_run()


def log_artifact(artifact_path, artifact_name=None):
    if _nolog:
        return

    try:
        mlflow.log_artifact(artifact_path, artifact_name)
    except Exception as e:
        logger.error("CANNOT LOG ARTIFACT")
        logger.debug("CANNOT LOG ARTIFACT %s: %s", artifact_path, e)


def log_params(params_dict):
    if _nolog:
        return

    try:
        mlflow.log_params(params_dict)
    except Exception as e:
        logger.error("CANNOT LOG PARAMS")
        logger.debug("CANNOT LOG PARAMS %s: %s", params_dict, e)


def log_param(name, value):
    if _nolog:
        return

    try:
        mlflow.log_param(name, value)
    except Exception as e:
        logger.error("CANNOT LOG PARAM")
        logger.debug("CANNOT LOG PARAM %s: %s", value, e)


def log_metric(name, value, step=0):
    if _nolog:
        return

    try:
        mlflow.log_metric(name, value, step=step)
    except Exception as e:
        logger.error("CANNOT LOG METRIC")
        logger.debug("CANNOT LOG METRIC %s: %s", value, e)


def log_model(model, name, registered_model_name, signature=None):
    if _nolog:
        return

    try:
        mlflow.pytorch.log_model(
            model,
            name,
            registered_model_name=registered_model_name,
        )
    except Exception as e:
        logger.error("CANNOT LOG MODEL")
        logger.debug("CANNOT LOG MODEL: %s", e)


def log_input(dataset, source, context="training", name=None):
    if _nolog:
        return

    dt = mlflow.data.from_numpy(dataset.numpy(), source=source, name=name)
    try:
        mlflow.log_input(
            dt,
            context=context,
        )
    except Exception as e:
        logger.error("CANNOT LOG INPUT DATASET")
        logger.error("CANNOT LOG INPUT DATASET: %s\n%s", e, dataset)


def load_model(run_id):
    logged_model = f"runs:/{run_id}/model"
    return mlflow.pytorch.load_model(logged_model)
