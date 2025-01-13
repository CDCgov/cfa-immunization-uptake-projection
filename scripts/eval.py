import iup.eval


def function_name(incident_test_data, incident_projections, score_funs):
    for score_fun in score_funs:
        score = iup.eval.score(incident_test_data, incident_projections, score_fun)
        # save these scores somewhere
        print(score)
