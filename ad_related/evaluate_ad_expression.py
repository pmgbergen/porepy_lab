# Below, the function I wrote for rapid evaluation of ad expressions. I've tried to make the documentation as
# clear as possible. It's far from perfect, but it will serve as a starting point for discussing if a functionality 
# like this one deserves to be promoted to the core.

def eval_ad_expression(
        ad_expression: pp.ad.Operator,
        grid_bucket: pp.GridBucket,
        dof_manager: pp.DofManager,
        name: str = None,
        print_expression: bool = False) -> tuple:

    """
    Utility function for rapid evaluation of ad expressions.

    NOTE: 
    -----
    This function is meant mostly for debugging purposes, use it with caution otherwise.

    PARAMETERS:
    ----------
    ad_expression: pp.ad.Operator
        ad expression to be evaluated. Note that if atomic ad variables are passed, it will result in an error.
    grid_bucket: pp.GridBucket
        Grid bucket containing the mixed-dimensional geometry and necessary discretization matrices in the respective.
        data dictionaries
    dof_manager: pp.DofManager
        Degree of freedom manager of the problem.
    name: (optional) String
        Name of the ad expression passed by the user. Default is None.
    print_expression: (optional) Bool
       Prints the value and the Jacobian of the resulting ad expression in the console. Default is False.

    RAISES:
    ------
    TypeError:
        If ad_expression is not of the type: pp.ad.Operator

    RETURNS:
    --------
    expression_num.val: np.nd_array
        Values of the evaluated expression
    expression_num.jac : sps.spmatrix
        Jacobian of the evaluated expression
    """

    # Sanity check: Make sure ad expression is of the type pp.ad.Operator
    # In particular, we want to avoid evaluating atomic ad variables and numpy-like objects
    if not isinstance(ad_expression, pp.ad.Operator):
        raise TypeError("Ad expression can only be of the type pp.ad.Operator")

    # Evaluate ad expression
    expression_eval = pp.ad.Expression(ad_expression, dof_manager)

    # Discretize ad expression
    expression_eval.discretize(grid_bucket)

    # Parse expression to retrieve the numerical values
    expression_num = expression_eval.to_ad(grid_bucket)

    # Print if necessary: Meant only for small arrays and matrices, a.k.a. debugging.
    if print_expression:
        if name is None:
            print('Evaluation of ad expression: \n')
            print(f'Array with values: \n {expression_num.val} \n')
            print(f'Jacobian with values: \n {expression_num.jac.A} \n')
        else:
            print(f'Evaluation of ad expression: {name} \n')
            print(f'Array with values: \n {expression_num.val} \n')
            print(f'Jacobian with values: \n {expression_num.jac.A} \n')

    return expression_num.val, expression_num.jac
