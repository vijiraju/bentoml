import bentoml

iris_clf_runner=bentoml.sklearn.get("iris_clf:latest").to_runner()
iris_clf_runner.init_local()
print(iris_clf_runner.predict.run([[6,2,8,2]]))