                for score_fun in score_funs:
                    score = eval.score(
                        incident_test_data, incident_projections, score_fun
                    )
                    print(f"{model=} {forecast_date=} {score_fun=} {score=}")
                    # save these scores somewhere
