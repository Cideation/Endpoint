# runner_net.py
import sfde_utility_foundation_extended as formulas

def run_affinity_net(input_dict, target_affinity):
    results = {}
    for name in dir(formulas):
        obj = getattr(formulas, name)
        if callable(obj) and getattr(obj, 'data_affinity', None) == target_affinity:
            try:
                required_args = obj.__code__.co_varnames[:obj.__code__.co_argcount]
                kwargs = {k: v for k, v in input_dict.items() if k in required_args}
                if len(kwargs) == len(required_args):
                    results[name] = obj(**kwargs)
            except Exception as e:
                results[name] = f"Error: {str(e)}"
    return results

# Example usage
if __name__ == "__main__":
    sample_data = {
        "quantity": 100,
        "unit_price": 50,
        "power_kw": 3,
        "hours": 8
    }
    print(run_affinity_net(sample_data, target_affinity="cost"))
    print(run_affinity_net(sample_data, target_affinity="energy"))
