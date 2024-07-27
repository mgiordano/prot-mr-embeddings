from prefect import task
from functools import wraps

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
                return task(func, **task_kwargs)(*args, **kwargs)  # Apply with kwargs
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator