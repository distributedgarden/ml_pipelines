import boto3
import datetime
import sagemaker
import os

from sagemaker.session import Session
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import ModelMetrics
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.experiments import Experiment
from smexperiments.trial import Trial


def fetch_ecr_image_uri(repository_name: str, region: str):
    """
    Description:
        - fetch the latest image URI from an ECR repository

    Args:
        - repository_name: ECR repository name
        - region: AWS region
    """
    ecr_client = boto3.client("ecr", region_name=region)
    try:
        response = ecr_client.describe_images(
            repositoryName=repository_name, maxResults=1
        )
        image_details = response["imageDetails"][0]
        image_digest = image_details["imageDigest"]

        return f"{repository_name}@{image_digest}"

    except Exception as error:
        print(f"Error fetching ECR image URI: {error}")
        return None


def create_sagemaker_experiment(base_name, sagemaker_session):
    """Create a SageMaker experiment with a unique name."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    unique_name = f"{base_name}-{timestamp}"
    return Experiment.create(
        experiment_name=unique_name,
        description="Experiment to fine-tune BERT model",
        sagemaker_session=sagemaker_session,
    )


def create_sagemaker_trial(experiment_name):
    """Create a trial for the experiment."""
    trial_name = f"MyBERTTrial-{sagemaker.utils.sagemaker_timestamp()}"
    return Trial.create(
        trial_name=trial_name,
        experiment_name=experiment_name,
    )


def setup_pytorch_estimator(image_uri, sagemaker_session, role_arn):
    """Set up a PyTorch estimator for training using a custom ECR image."""
    return PyTorch(
        entry_point="train.py",
        #role=sagemaker_session.get_caller_identity_arn(),
        role=role_arn,
        image_uri=image_uri,
        instance_count=1,
        instance_type="ml.m5.large",
        sagemaker_session=sagemaker_session,
    )


def setup_training_step(estimator):
    """Set up the training step for the pipeline."""
    return TrainingStep(
        name="BERTModelTraining",
        estimator=estimator,
        inputs={"training": TrainingInput(s3_data="s3://imdb-content/train.csv")},
    )


def setup_model_registration_step(estimator, training_step):
    """Set up the model registration step for the pipeline."""
    model_metrics = ModelMetrics(model_statistics={})
    return RegisterModel(
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


def main():
    """Main function to execute the SageMaker pipeline."""
    # aws_region = "us-east-1"
    aws_region = os.getenv("AWS_REGION", "us-east-1")
    role_arn = os.getenv("AWS_SAGEMAKER_ER_ARN")

    if not role_arn:
        raise ValueError("SageMaker execution role ARN is required")

    boto_session = boto3.Session(region_name=aws_region)
    sagemaker_session = Session(boto_session=boto_session)

    repo_name = "sagemaker-ml-pipelines"
    image_uri = fetch_ecr_image_uri(repo_name, aws_region)

    print(f"\n>>>>image_uri: {image_uri}")

    if not image_uri:
        raise RuntimeError("Failed to fetch ECR image URI.")

    experiment = create_sagemaker_experiment("MyBERTExperiment", sagemaker_session)
    trial = create_sagemaker_trial(experiment.experiment_name)

    estimator = setup_pytorch_estimator(image_uri, sagemaker_session, role_arn)
    training_step = setup_training_step(estimator)
    registration_step = setup_model_registration_step(estimator, training_step)

    pipeline = Pipeline(
        name="BERT-Training-Pipeline",
        steps=[training_step, registration_step],
        sagemaker_session=sagemaker_session,
    )

    pipeline.upsert(role_arn=role_arn)
    execution = pipeline.start()
    print(f"Pipeline execution started with ARN: {execution.arn}")


if __name__ == "__main__":
    main()
