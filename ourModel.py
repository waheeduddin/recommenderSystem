import pandas as pd
import graphlab
import myRESTimplementation
# pass in column names for each CSV and read them using pandas.
# Column names available in the readme file

#Reading users file:
u_cols = ['user_id', 'age', 'gender', 'Flight type', 'phone Type' , 'environment','departure location']
users = pd.read_csv('brick/some.user', sep='|', names=u_cols,
 encoding='latin-1')

#Reading items file:
p_cols = ['product_id', 'price' ,'brand','luxury', 'presents', 'food', 'fashion', 'accessories']
products = pd.read_csv('brick/file.prod', sep='|', names=p_cols,
 encoding='latin-1')

#ratings file training
r_cols = ['user_id', 'environment','Flight type','product_id','departure location','gender','age', 'phone Type']
ratings_base = pd.read_csv('brick/our.base', sep='|', names=r_cols,
                           encoding='latin-1')

#ratings file test
ratings_test = pd.read_csv('brick/our.test', sep='|', names=r_cols, encoding='latin-1')
df = ratings_test.drop(['product_id'] , 1)

#converting data to frames for faster processing
myUsers_data = graphlab.SFrame(users)
myProducts_data = graphlab.SFrame(products)
train_data = graphlab.SFrame(ratings_base)
test_data = graphlab.SFrame(df)

#test_data.print_rows(num_rows=100)
# #Train Model#
item_sim_model = graphlab.recommender.create(train_data, user_id='user_id', item_id='product_id',user_data = myUsers_data, item_data = myProducts_data, ranking = True)

#get recommendations
item_sim_recomm = item_sim_model.recommend(test_data,k = 5) # the user id and the number of predictions needs to come from backend

#print results
item_sim_recomm.export_csv('our.results' , delimiter='|', line_terminator='\n', header=False)
