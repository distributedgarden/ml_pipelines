import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep, RegisterModel
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.pytorch import PyTorch
from sagemaker.experiments import Experiment, Trial
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import ModelMetrics
from sagemaker.workflow.step_collections import RegisterModel


def get_sagemaker_pipeline():
    # create an experiment to track training lineage
    experiment = Experiment.create(
        experiment_name="MyBERTExperiment",
        description="Experiment to fine-tune BERT model",
    )

    trial_name = "MyBERTTrial-" + sagemaker.utils.sagemaker_timestamp()
    trial = Trial.create(
        trial_name=trial_name,
        experiment_name=experiment.experiment_name,
    )

    # define the estimator
    pytorch_estimator = PyTorch(
        entry_point="training/train.py",
        role=sagemaker.get_execution_role(),
        framework_version="1.8.1",
        py_version="py3",
        instance_count=1,
        instance_type="ml.m5.large",
    )

    # define the training step
    training_step = TrainingStep(
        name="BERTModelTraining",
        estimator=pytorch_estimator,
        inputs={"training": TrainingInput(s3_data="s3://path-to-your-dataset/")},
        experiment_config={
            "TrialName": trial.trial_name,
            "TrialComponentDisplayName": "Training",
        },
    )

    # define model metrics
    model_metrics = ModelMetrics(
        model_statistics={
            # Add your model evaluation metrics here
            # Example: "Accuracy": {"value": 0.8, "standard_deviation": 0.01}
        }
    )

    # define the model registry
    register_step = RegisterModel(
        name="RegisterBERTModel",
        estimator=pytorch_estimator,
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.t2.medium", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name="BERTModelPackageGroup",
        model_metrics=model_metrics,
    )

    # define the pipeline
    pipeline = Pipeline(
        name="BERT-Training-Pipeline",
        steps=[training_step, register_step],
        sagemaker_session=PipelineSession(),
    )

    return pipeline


pipeline = get_sagemaker_pipeline()
pipeline.upsert(role_arn=sagemaker.get_execution_role())
