import kfp
from kfp import dsl
from kfp.dsl import Input, InputPath, Output, OutputPath, Dataset, component


@dsl.container_component
def data_processing_op():
    return dsl.ContainerSpec(
        image='dataguru97/ml-mlops-appa:latest',
        command=["python","src/data_processing.py"]
    )


@dsl.container_component
def model_training_op():
    return dsl.ContainerSpec(
        image='dataguru97/ml-mlops-appa:latest',
        command=["python","src/model_training.py"]
    )
    
@dsl.pipeline(name="MLOps Pipeline", description="This a Cancer Prediction Pipeline")
def mlops_pipeline():
    data_processing_op_ = data_processing_op()
    model_training_op_ = model_training_op().after(data_processing_op_)

if __name__=="__main__":
    kfp.compiler.Compiler().compile(
        mlops_pipeline,
        "mlops_pipeline.yaml"
    )
    