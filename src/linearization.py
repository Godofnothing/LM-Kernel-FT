from functorch import jvp


__all__ = ["linearize"]


def _sub(x, y):
  subs = []
  for i in range(len(x)): 
    subs.append(x[i]-y[i])
  return tuple(subs)

def _add(x, y):
  adds = []
  if x is None or y is None:
    return None
  for i in range(len(x)): 
    adds.append(x[i]+y[i])
  return tuple(adds)

def linearize(f, params, start: str):
  def f_lin(p, *args, **kwargs):
      dparams = _sub(p, params)
      f_params_x, proj = jvp(lambda param: f(param, *args, **kwargs),
                            (params,), (dparams,))
      if start == 'ref_start': 
        return _add(f_params_x, proj)
      elif start == 'zero_start': 
        return tuple(proj)
      else: 
        raise ValueError("Unspecified Initialization!")   
  return f_lin