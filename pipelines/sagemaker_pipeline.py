import boto3
import sagemaker

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.pytorch import PyTorch
from sagemaker.experiments import Experiment, Trial
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import ModelMetrics
from sagemaker.workflow.step_collections import RegisterModel


def get_latest_ecr_image_uri(repository_name, aws_region):
    """
    Description:
        - fetch the latest image URI from an ECR repository
    """
    ecr_client = boto3.client("ecr", region_name=aws_region)
    try:
        response = ecr_client.describe_images(
            repositoryName=repository_name, maxResults=1
        )
        image_details = response["imageDetails"][0]
        image_digest = image_details["imageDigest"]
        image_uri = f"{repository_name}@{image_digest}"

        return image_uri

    except Exception as e:
        print(f"Error fetching image URI: {e}")

        return None


def create_experiment(name):
    """
    Description:
        - create a SageMaker experiment
        - experiments will enable tracking metadata for each training job
    """
    experiment = Experiment.create(
        experiment_name=name, description="Experiment to fine-tune BERT model"
    )

    return experiment


def create_trial(experiment_name):
    """
    Description:
        - create a trial for the experiment.
    """
    trial_name = f"MyBERTTrial-{sagemaker.utils.sagemaker_timestamp()}"
    trial = Trial.create(trial_name=trial_name, experiment_name=experiment_name)

    return trial


def create_pytorch_estimator(ecr_image_uri):
    """
    Description:
        - estimator for training using a custom ECR image
    """
    estimator = PyTorch(
        entry_point="train.py",
        role=sagemaker.get_execution_role(),
        image_uri=ecr_image_uri,
        instance_count=1,
        instance_type="ml.m5.large",
    )

    return estimator


def create_training_step(estimator, trial_name):
    """
    Description:
        - step: training
    """
    step = TrainingStep(
        name="BERTModelTraining",
        estimator=estimator,
        inputs={"training": TrainingInput(s3_data="s3://imdb-content/train.csv")},
        experiment_config={
            "TrialName": trial_name,
            "TrialComponentDisplayName": "Training",
        },
    )

    return step


def create_register_model_step(estimator, training_step):
    """
    Description:
        - step: register the model
        - SageMaker Model Registry
    """
    model_metrics = ModelMetrics(
        model_statistics={
            # Example: "Accuracy": {"value": 0.8, "standard_deviation": 0.01}
        }
    )

    step = RegisterModel(
        name="RegisterBERTModel",
        estimator=estimator,
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name="BERTModelPackageGroup",
        model_metrics=model_metrics,
    )

    return step


def main():
    """
    Description:
        - run pipeline
    """
    ecr_repository_name = "sagemaker-ml-pipelines"
    aws_region = "us-east-1"
    ecr_image_uri = get_latest_ecr_image_uri(ecr_repository_name, aws_region)

    if not ecr_image_uri:
        raise RuntimeError("ECR image URI could not be fetched")

    experiment = create_experiment("MyBERTExperiment")
    trial = create_trial(experiment.experiment_name)

    pytorch_estimator = create_pytorch_estimator(ecr_image_uri)
    training_step = create_training_step(pytorch_estimator, trial.trial_name)
    register_step = create_register_model_step(pytorch_estimator, training_step)

    pipeline = Pipeline(
        name="BERT-Training-Pipeline",
        steps=[training_step, register_step],
        sagemaker_session=PipelineSession(),
    )

    pipeline.upsert(role_arn=sagemaker.get_execution_role())
    execution = pipeline.start()
    print(f"Pipeline execution started with ARN: {execution.arn}")


if __name__ == "__main__":
    main()
