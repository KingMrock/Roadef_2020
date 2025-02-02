"""Solver for RTE ROADEF Challenge 2020 by Marco Langiu."""
import time
t0 = time.time()

import contextlib
try:
    import ujson as json
except ImportError:
    try:
        import simplejson as json
    except ImportError:
        import json
from math import ceil
import os
import sys
from gurobipy import Model, GRB, LinExpr

from operator import itemgetter
item1 = itemgetter(1)

from collections import defaultdict

from numpy import full


@contextlib.contextmanager
def redirect_output(to=os.devnull, append=False):
    """Execute code and redirect output to a file or stream."""
    with open(to, 'a' if append else 'w') as f:
        save_stdout = sys.stdout
        sys.stdout = f
        yield
        sys.stdout = save_stdout


def printmsg(msg):
    """Print a message."""
    print()
    print('=' * len(msg))
    print(msg)
    print('=' * len(msg))
    print()


def read_json(filename: str):
    """Read json file and normalize instance data."""
    t0 = time.time()
    with open(filename, 'r') as f:
        instance = json.load(f)

    instance['Seasons'] = {season: [int(t) for t in timesteps]
                           for season, timesteps
                           in instance['Seasons'].items()}

    # Normalize input
    for intervention in instance['Interventions'].values():
        intervention['tmax'] = int(intervention['tmax'])
        intervention['Delta'] = [int(delta) for delta in intervention['Delta']]
        intervention['workload'] = {r: {int(t): {int(ts): wl
                                                 for ts, wl in tswl.items()}
                                        for t, tswl in workload_r.items()}
                                    for r, workload_r
                                    in intervention['workload'].items()}
        intervention['risk'] = {int(t): {int(ts): risk_ts
                                         for ts, risk_ts in tsrisk.items()}
                                for t, tsrisk in intervention['risk'].items()}

    return instance, time.time() - t0


def find_earliest_start(t_goal, deltas):
    """Find the earliest start time with an active range reaching t_goal.

    Example
    -------
        t_goal == 5
        delta[t_goal - 1] == 3
    We first take a guess...

        |√-2|√-1| √ |
          <------3
    * ... Case 1 (e.g. deltas = [4, 4, 3, 3, 3]): guess DOES reach √
          |√-2|√-1| √ |
            3------>
      * Does the one before also reach √?
            |√-3|√-2|√-1| √ |
              4---------->
      * Yes! Does the one before also reach √?
            |√-4|√-3|√-2|√-1| √ |
              4---------->
        No! so return √-3
    * ... Case 2 (e.g. deltas = [2, 2, 2, 3, 3]): guess does NOT reach √
          |√-2|√-1| √ |
            2-->
      * Does the one after reach √?
            |√-2|√-1| √ |√+1|
                  3------>
        Yes! So return √-1
    * ... Case 3 (e.g. deltas = [4, 4, 2, 3, 3]): will incorrectly return √-1!
    """
    # t_guess = ti + 1
    ti = max(0, t_goal - deltas[t_goal - 1])
    if ti + deltas[ti] >= t_goal:
        # starting at t_guess reaches t_goal, reducing time until it doesn't...
        ti -= 1
        while ti > 0 and ti + deltas[ti] >= t_goal:
            ti -= 1
        return ti + 2  # ti is the first index for which t_goal is NOT reached
    else:  # t_guess does not reach t_goal, increase time until it does...
        ti += 1
        while ti + deltas[ti] < t_goal:
            ti += 1
        return ti + 1


def find_earliest_start_after(t_goal, deltas, t_guess):
    """Find the earliest start time with an active range reaching t_goal.

    We only need to advance
        |√-3|√-2|√-1| √ |
         a:3----->
    Next?
        |√-3|√-2|√-1| √ |
          a   3------>
    Return √-2
    """
    ti = t_guess - 1
    while ti + deltas[ti] < t_goal:
        ti += 1
    return ti + 1


class SparseBuilder:
    """Helper class for building a sparse matrix incrementally."""

    def __init__(self):
        self._names = []
        self._indices = []  # variable indices
        self._coeffs = []
        self._indptr = [0]  # number of cumulative nonzero values for each row
        self._sense = []
        self._rhs = []

    def add_row(self, name, coeffs, sense, rhs):
        """Add a row to the sparse matrix."""
        self._names.append(name)
        self._indices.extend(coeffs.keys())
        self._coeffs.extend(coeffs.values())
        self._indptr.append(self._indptr[-1] + len(coeffs.values()))
        self._sense.append(sense)
        self._rhs.append(rhs)

    @property
    def A(self):
        """Get the sparse matrix in csr format."""
        from scipy.sparse import csr_matrix
        return csr_matrix((self._coeffs, self._indices, self._indptr))

    @property
    def sense(self):
        """Get the list of sense chars ('<', '=', or '>')."""
        return self._sense

    @property
    def b(self):
        """Get the RHS value."""
        return self._rhs

    @property
    def names(self):
        """Get the names of constraints."""
        return self._names


class Timer:
    """Helper class for timing."""

    def __init__(self, t0=None):
        if t0 is None:
            self.t0 = time.time()
        else:
            self.t0 = t0

    def timeit(self, msg='', tbefore=None):
        """Calculate time that has passed since tbefore and print msg."""
        t = time.time()
        print(msg, t - (self.t0 if tbefore is None else tbefore), 's')
        return t


class InstanceModel(Model):
    """An extension of the GUROBI model for the ROADEF2020 problem."""

    def __setattr__(self, attr, value):
        """Let GUROBI handle its own Model attributes but allow new ones."""
        if hasattr(Model, attr):
            super().__setattr__(attr, value)
        else:
            object.__setattr__(self, attr, value)

    def __init__(self, name, resources, seasons, interventions, exclusions, T,
                 n_scenarios, tau, req_robustness, alpha, start_by=False,
                 approach='indicator', dominance=None):
        super().__init__(name)
        self.name = name
        self.resources = resources
        self.seasons = seasons
        self.interventions = interventions
        self.exclusions = exclusions
        self.T = T
        self.n_scenarios = n_scenarios
        self.tau = tau
        self.req_robustness = req_robustness
        self.alpha = alpha
        self.H = range(1, T+1)
        self.start_by = start_by
        self.approach = approach.lower()

        self.nstart = sum(i_data['tmax'] for i_data in interventions.values())
        self.timer = timer = Timer(time.time())

        # Helper object to buid constraint matrix for start variables
        sb = SparseBuilder()

        # scenario-specific overall risk: [ti -> [si -> {var_idx: coeff}]]
        scenario_coeffs = [[dict() for _ in range(n_scenarios_t)]
                           for n_scenarios_t in n_scenarios]

        # coefficients for overall mean risk: [ti -> {var_idx: coeff}]
        mean_risk_coeffs = [dict() for _ in range(T)]  # TEMP
        # coefficients for resource constraints: {r: [t -> {var_idx: coeff}]}
        resources_coeffs = {r: [defaultdict(float) for t in self.H]
                            for r in resources}
        # individual tau quantiles: [ti -> {var_idx: coeff}]
        individual_quantile_coeffs = [dict() for _ in range(T)]
        # objective coefficients for start variables in the heuristic and final
        # formulations: [var_idx -> coeff]
        start_coeffs = full(self.nstart, 0.0)

        # NOTE: Objective coefficients are combinations of the two objective
        #       parts. For each time step, the first is the overall mean risk:
        #           obj1(t) = OMR(t)
        #       The second is the excess, which can be expressed as the
        #       difference between the tau-quantile and the mean risk, i.e:
        #           obj2(t) = E(t) = max(0, Q(t) - OMR(t))
        #       The two parts are weighted with alpha and (1 - alpha)
        #       respectively, giving:
        #           obj(t) = alpha * obj1(t) + (1 - alpha) * obj2(t)
        #       As the total objective consists of the time-weighted average,
        #       the above coefficients further need to be divided by T to
        #       obtain the variable-specific coefficients.
        obj1_factor = (alpha) / T
        obj2_factor = (1 - alpha) / T

        tm = timer.timeit(f't = {time.time() - t0:.3f} s, '
                          'setting up data structures took:')

        # Prepare index ranges for start variables
        curr = 0
        start_ranges = {}
        for i, i_data in interventions.items():
            prev = curr
            curr += i_data['tmax']
            start_ranges[i] = i_start = range(prev, curr)

            # Enforces "interventions are scheduled once" (4.1.2)
            sb.add_row(f'schedule_{i}',
                       coeffs={i_start_t: 1 for i_start_t in i_start},
                       sense='=',
                       rhs=1)

            # Collect coefficients for resource constraints
            for r, workload in i_data['workload'].items():
                r_coeffs = resources_coeffs[r]
                for t, tswl in workload.items():
                    r_t_coeffs = r_coeffs[t - 1]
                    for ts, wl in tswl.items():
                        r_t_coeffs[i_start[ts - 1]] += wl

            # Collect coefficients for risk constraints and objective
            for t, risk_coeffs in i_data['risk'].items():
                ti = t - 1
                scenario_coeffs_t = scenario_coeffs[ti]
                mean_risk_coeffs_t = mean_risk_coeffs[ti]
                individual_quantile_coeffs_t = individual_quantile_coeffs[ti]
                scenario_factor = 1 / n_scenarios[ti]
                for ts, coeffs in risk_coeffs.items():
                    i_start_t = i_start[ts - 1]
                    # TODO: cache sorting?
                    individual_quantile_coeffs_t[i_start_t] \
                        = sorted(enumerate(coeffs),
                                 key=item1)[req_robustness[ti] - 1][1]
                    mean_risk_coeffs_t[i_start_t] = mean_risk_ts \
                        = sum(coeffs) * scenario_factor
                    start_coeffs[i_start_t] += mean_risk_ts * obj1_factor
                    for si, coeff in enumerate(coeffs):
                        scenario_coeffs_t[si][i_start_t] = coeff

        # NOTE: For the heuristic we assume Q(t) >= OMR(t) resulting in:
        #           obj(t) = alpha * OMR(t) + (1 - alpha) * (Q(t) - OMR(t))
        #                  = (2 * alpha - 1) * OMR(t) + (1 - alpha) * Q(t)
        #       That means coefficients of the start variables are scaled by
        #       (2 - 1 / alpha).
        #       Additionally, the heuristic formulation uses the individual
        #       quantile as a surrogate for the actual overall quantile, hence
        #       the coefficients of the corresponding start variables are
        #       increased by the risk values from the scenario that dominates
        #       the individual quantile.
        #       This allows the heuristic formulation to only use the start
        #       variables!
        heuristic_coeffs = start_coeffs * (2 - 1 / alpha)
        for individual_quantile_coeffs_t in individual_quantile_coeffs:
            for var, coeff in individual_quantile_coeffs_t.items():
                heuristic_coeffs[var] += coeff * obj2_factor

        self.scenario_coeffs = scenario_coeffs  # for dominance constrs
        self.mean_risk_coeffs = mean_risk_coeffs  # for excess constrs
        self.start_coeffs = start_coeffs  # for update / SOS constrs
        self.start_ranges = start_ranges  # for SOS constrs / writing
        tm = timer.timeit(f't = {time.time() - t0:.3f}, '
                          'collecting coefficients took:', tm)

        # Defining rows for resource constraints
        for r, r_coeffs in resources_coeffs.items():
            for (ti, r_t_coeffs), lb, ub \
                in zip(enumerate(r_coeffs, 1),
                       resources[r]['min'],
                       resources[r]['max']):
                sb.add_row(f'{r}_{t}_lb', coeffs=r_t_coeffs, sense='>', rhs=lb)
                sb.add_row(f'{r}_{t}_ub', coeffs=r_t_coeffs, sense='<', rhs=ub)
        tm = timer.timeit(f't = {time.time() - t0:.3f}, '
                          'setting up resource constraints took:', tm)

        # Defining rows for exclusion constraints
        for e, (i1, i2, s) in self.exclusions.items():
            # Retrieve concerned interventions and data...
            intervention1 = interventions[i1]
            intervention2 = interventions[i2]
            i1_deltas = intervention1['Delta']
            i2_deltas = intervention2['Delta']
            t_max_1 = intervention1['tmax']
            t_max_2 = intervention2['tmax']
            i1_start = start_ranges[i1]
            i2_start = start_ranges[i2]
            season = self.seasons[s]

            i1_t0 = find_earliest_start(season[0], i1_deltas)
            i2_t0 = find_earliest_start(season[0], i2_deltas)

            for t in season:
                t1_start = find_earliest_start_after(t, i1_deltas, i1_t0) - 1
                t1_end = min(t_max_1, t)
                t2_start = find_earliest_start_after(t, i2_deltas, i2_t0) - 1
                t2_end = min(t_max_2, t)
                sb.add_row(f'{e}_{t}',
                           coeffs={var_idx: 1 for var_idx
                                   in (*i1_start[slice(t1_start, t1_end)],
                                       *i2_start[slice(t2_start, t2_end)])},
                           sense='<',
                           rhs=1)
        tm = timer.timeit(f't = {time.time() - t0:.3f}, '
                          'setting up exclusion constraints took:', tm)

        # Creation of start variables, initially with heuristic coefficents
        start_vars = self.addMVar(self.nstart, vtype=GRB.BINARY,
                                  obj=heuristic_coeffs, name='start_')
        self.start_vars = start_vars
        self.modelSense = GRB.MINIMIZE
        tm = timer.timeit(f't = {time.time() - t0:.3f}, '
                          'generating start vars took:', tm)

        # Adding main constraints
        self.start_cons = self.addMConstr(sb.A, start_vars, sb.sense, sb.b)
        timer.timeit(f't = {time.time() - t0:.3f}, '
                     'setting constraints took:', tm)

        # now optimize heuristic then update

    def update_heuristic(self):
        """Update the model from the heuristic formulation to the final one."""
        n_scenarios, tau, T, alpha = \
            self.n_scenarios, self.tau, self.T, self.alpha

        timer = self.timer
        tm = time.time()
        # Set start coefficients to the final values
        start_vars = self.start_vars
        start_vars.obj = self.start_coeffs

        tm = timer.timeit(f't = {time.time() - t0:.3f}, '
                          'updating objective coeffs took:', tm)
        # Variables for each scenario of each timestep, indicating whether the
        # tau quantile dominates the corresponding scenario's risk level.
        self.dominance = dominance = self.addMVar(sum(n_scenarios),
                                                  vtype=GRB.BINARY,
                                                  name='dominance_')
        dominance.BranchPriority = 2
        # Variables rsepresenting tau quantiles
        self.Q = Q = self.addVars(T, lb=0, name=f'{tau}_quantile_')
        # Variables for the excess over the overall mean risk
        self.E = E = self.addVars(T, lb=0, obj=(1 - alpha) / T, name='excess_')

        tm = timer.timeit(f't = {time.time() - t0:.3f}, '
                          'adding quantile, excess and dominance vars took:',
                          tm)

        # Helper object for for quantile, excess and dominance constraints
        # Variables are ordered as: [d1, ..., dN, Q1, ..., QT, E1, ..., ET]
        db = SparseBuilder()
        curr = 0
        scenario_coeffs = self.scenario_coeffs
        mean_risk_coeffs = self.mean_risk_coeffs
        req_robustness = self.req_robustness
        for ti, n_scenarios_t in enumerate(n_scenarios):
            prev = curr
            curr += n_scenarios_t
            t_dom_range = range(prev, curr)
            dominance_t = dominance[t_dom_range]
            dominance_t_list = dominance_t.tolist()
            Q_t = Q[ti]
            E_t = E[ti]
            # Ensure the required number of scenarios is dominated by Q[ti]
            db.add_row(f'dominated_scenarios_at_{ti}',
                       coeffs={var_idx: 1
                               for var_idx in dominance_t._idxarr()},
                       sense='>', rhs=req_robustness[ti])
            # Enforce excess is max(0, Q(t) - OMR(t)) via:
            #   Q(t) - OMR(t) <= E(t) -> OMR(t) -Q(t) + E(t) >= 0
            # NOTE: ti and T + ti are the indices of Q(t) and E(t)
            db.add_row(f'excess_at_{ti}',
                       coeffs={**mean_risk_coeffs[ti],
                               Q_t._colno: -1,
                               E_t._colno: 1},
                       sense='>', rhs=0)
            scenario_coeffs_t = scenario_coeffs[ti]
            # Add indicator constraints enforcing risk[ti][si] <= Q[ti]
            for si in range(n_scenarios_t):
                scenario_coeffs_t_s = scenario_coeffs_t[si]
                coeffs = [*scenario_coeffs_t_s.values()]
                coeffs.append(-1)
                vars = start_vars[[*scenario_coeffs_t_s.keys()]].tolist()
                vars.append(Q_t)
                self.addGenConstrIndicator(
                    binvar=dominance_t_list[si],
                    binval=True,
                    lhs=LinExpr(coeffs, vars),
                    sense='<',
                    rhs=0,
                    name=f'dominance_indication_{ti}_{si}'
                )
        tm = timer.timeit(f't = {time.time() - t0:.3f}, '
                          'adding indicator constraints took:', tm)

        vars = [*start_vars.tolist(), *dominance.tolist(),
                *Q.values(), *E.values()]
        self.dom_cons = self.addMConstr(db.A, vars, db.sense, db.b)

        timer.timeit(f't = {time.time() - t0:.3f}, '
                     'adding dominance constraints took:', tm)

    def add_start_sos(self):
        """Add SOS constraints with objective coefficients as weights."""
        start_vars = self.start_vars
        coeffs = -self.start_coeffs
        for start_range in self.start_ranges.values():
            self.addSOS(GRB.SOS_TYPE1,
                        start_vars[start_range].tolist(),
                        coeffs[start_range])

    def write_result(self, file=sys.stdout):
        """Print start times for every intervention."""
        start_vals = self.start_vars.x

        def entries():
            for i in self.interventions:
                start_vals_i = start_vals[self.start_ranges[i]]
                t = 1 + start_vals_i.argmax()
                yield f'{i} {t}\n'
        file.writelines(entries())

    def solve(self, **options):
        """Solve the GUROBI model."""
        t0 = time.time()
        self.resetParams()
        for option, value in options.items():
            self.setParam(option, value)
        self.optimize()
        return self.status, time.time() - t0


def solve(input_file_path, t_lim, name, output_file_path, seed, approach):
    """Create a GUROBI model from instance data."""

    def whatsleft():
        t_left = t_lim - (time.time() - t0)
        if t_left < 0:
            exit(0)
        return t_left

    instance, read_time = read_json(input_file_path)
    printmsg(f'Step 1 (reading input file) took: {read_time} s')

    resources = instance['Resources']
    seasons = instance['Seasons']
    interventions = instance['Interventions']
    # Filter exclusions for empty seasons
    exclusions = {e: exclusion
                  for e, exclusion in instance['Exclusions'].items()
                  if seasons[exclusion[-1]]}
    T = instance['T']
    n_scenarios = instance['Scenarios_number']
    tau = instance['Quantile']
    req_robustness = [ceil(tau * n_scenario_t) for n_scenario_t in n_scenarios]
    alpha = instance['Alpha']

    t1 = time.time()
    # with redirect_output():  # to silence License statement
    gm = InstanceModel(name, resources, seasons, interventions, exclusions,
                       T, n_scenarios, tau, req_robustness, alpha,
                       approach='heuristic')
    print('Generating model took', time.time() - t1, 's,', whatsleft(), 'left')

    # General options
    options = dict(
        Seed=seed,
        threads=4,
        NumericFocus=1,  # TODO: often large coefficient ranges!
        Heuristics=0.95,  # We're interested in good solutions
        MIPFocus=1,  # focus on good quality feasible solutions
        BranchDir=1,  # up branch first
        Method=1  # 1=dual simplex
    )

    out_dir = os.path.dirname(output_file_path)
    with redirect_output(os.path.join(out_dir, f'{name}_heuristic.log')):
        gm.solve(**options, MipGap=0.01, Presolve=0, TimeLimit=whatsleft()/0.9)
    try:
        tw0 = time.time()
        with open(output_file_path, 'w') as f:
            gm.write_result(f)
        tw = time.time() - tw0
    except Exception as e:
        printmsg(f"Couldn't write heuristic solution! {e}")
        tw = 10

    t2 = time.time()
    printmsg(f'Step 2 (solving heuristic) took {t2 - t1} s')

    gm.update_heuristic()
    gm.add_start_sos()  # weights based on objective coefficients

    logfile = os.path.join(out_dir, f'{name}.log')

    t_update = time.time() - t2
    print('Updating model took', t_update, 's,', whatsleft(), 'left')

    with redirect_output(logfile):
        t_left = max(0, whatsleft() - 2 * t_update)
        gm.solve(**options, MipGap=0, TimeLimit=t_left,
                 NoRelHeurTime=t_left * 0.05,
                 ImproveStartTime=t_left * 0.95,
                 Cuts=3  # very aggressive cut generation
                 )

    if gm.status == 9:  # Terminated due to time limit
        try:
            with open(output_file_path, 'w') as f:
                gm.write_result(f)
        except Exception as e:
            printmsg(f"Couldn't write solution! {e}")

        with redirect_output(logfile, append=True):
            gm.solve(**options, MipGap=0, ImproveStartTime=0,
                     TimeLimit=max(0, whatsleft() - 3 * tw))

    try:
        with open(output_file_path, 'w') as f:
            gm.write_result(f)
    except Exception as e:
        printmsg(f"Couldn't write solution! {e}")

    t3 = time.time()
    printmsg(f'Step 3 (solving main problem) took {t3 - t2} s')

    # Do some fine tuning if it makes sense
    if whatsleft() > 3 * tw:
        EPS = 1e-09
        with redirect_output(logfile, append=True):
            gm.solve(**options, MipGap=0, IntFeasTol=EPS, FeasibilityTol=EPS,
                     OptimalityTol=EPS, TimeLimit=max(0, whatsleft() - 2 * tw))

        try:
            with open(output_file_path, 'w') as f:
                gm.write_result(f)
        except AttributeError as e:
            printmsg(f"Couldn't write solution! {e}")

        t4 = time.time()
        printmsg(f'Step 4 (solving fine-tuning problem) took {t4 - t3} s')

    return gm


if __name__ == '__main__':
    """Create and solve a mathematical program based on instance data.

    Parameters
    ----------
    -t time_limit_in_s
    -p path/to/instance_data.json
    -o path/to/instance_output.txt
    -name
    -s seed
    -a approach
    """
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-t", required=False, default=900,
                    type=float,
                    help="Total wall time in s")
    ap.add_argument("-p", required=False, default=None,
                    type=str, help="Path to the instance data file")
    ap.add_argument("-o", required=False, default=None, type=str,
                    help="Path to result file, if none is given, the filename"
                    "of the input data file and the extension `.txt' is used,"
                    "and the file is created in the working directory.")
    ap.add_argument("-name", required=False, action='store_true',
                    help="Print teamID")
    ap.add_argument("-s", required=False, default=42, type=int,
                    help="Random seed")
    ap.add_argument("-a", required=False, default='indicator', type=str,
                    help="Solution approach")

    args = vars(ap.parse_args())
    name = args['name']
    t_lim = args['t']
    input_file_path = args['p']
    output_file_path = args['o']
    seed = args['s']
    approach = args['a']
    if name:
        print('J73')
        if t_lim == 900 and input_file_path is output_file_path is None \
                and seed == 42:
            sys.exit()

    if input_file_path is None:
        print("Please provide a path to an instance data file using the `-p'"
              'flag!')
        sys.exit()
    import os
    if output_file_path is None:
        output_path, filename = os.path.split(input_file_path)
        name = os.path.splitext(filename)[0]
        output_file_path = os.path.join(output_path, f'{name}.txt')
    else:
        output_path, filename = os.path.split(output_file_path)
        name = os.path.splitext(filename)[0]

    solve(input_file_path, t_lim, name, output_file_path, seed, approach)
