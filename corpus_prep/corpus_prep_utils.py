from prefect import task as prefect_task
from functools import wraps
from dataclasses import dataclass, field

@dataclass
class MRFilter:

    by: str = field(metadata={"description": "The field or special logic by which to filter"}, default="none")
    name: str = field(metadata={"description": "The name of the filter"}, default="Unnamed filter")
    condition: str = field(metadata={"description": "The condition operator by which to filter"}, default="")
    value: str = field(metadata={"description": "The value by which to filter the condition"}, default="")

def in_prefect_flow_context():
    try:
        from prefect import context
        flow_run_ctx = context.get_run_context()
        return (
            flow_run_ctx.flow_run is not None
        )  # Check if in a flow run context
    except (ImportError, AttributeError, RuntimeError):  
        return False
    
# custom task decorator to switch prefect behaviour to normal
# when calling from outside flow
def task(**task_kwargs): # Accept kwargs for @task
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if in_prefect_flow_context():
                return prefect_task(func, **task_kwargs)(*args, **kwargs)  # Apply with kwargs
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator

# #######################################
#      CONSTANTS                        #
# #######################################

class dataset_names():
    TEST_GROUP = "testGroupDataset"
    NANO_GROUP = "nanoGroupDataset"

class filters():
    MR_FILTER_NONE = MRFilter(by="none", name="filter_none")

    MR_FILTER_KEEP_SIGNIFICANT = MRFilter(condition="include", by="significance", name="filter_keep_significant")

    MR_FILTER_DROP_SMR = MRFilter(condition="!=", by="type", value="SMR", name="filter_drop_smr")

    MR_FILTER_DROP_NE = MRFilter(condition="!=", by="type", value="NE", name="filter_drop_ne")

    MR_FILTER_DROP_NN = MRFilter(condition="!=", by="type", value="NN", name="filter_drop_nn")

    MR_FILTER_KEEP_LEN6PLUS = MRFilter(condition= ">=", by="length", value=6, name="filter_keep_length_6plus")

class partition_rules():
    PARTITION_RULE_USE_ALL = {
        "name" : "partition_use_all"
    }