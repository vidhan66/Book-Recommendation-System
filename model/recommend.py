from fastapi import HTTPException, Query
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import zipfile
import os

CBF_data = pd.read_csv("CBF_data.csv")
CF_data1 = pd.read_csv("CF_data1.csv")
CF_data2 = pd.read_csv("CF_data2.csv")

if "Title-Author" not in CF_data1.columns:
    CF_data1['Title-Author'] = CF_data1['Book-Title'] + " by " + CF_data1['Book-Author']
if "Title-Author" not in CF_data2.columns:
    CF_data2['Title-Author'] = CF_data2['Book-Title'] + " by " + CF_data2['Book-Author']

CF_data1 = CF_data1.drop_duplicates(subset='Title-Author')
CF_data2 = CF_data2.drop_duplicates(subset='Title-Author')

exp_user_matrix = CF_data1.pivot(index='User-ID', columns='Title-Author', values='Book-Rating').fillna(0)
active_user_matrix = CF_data2.pivot(index='User-ID', columns='Title-Author', values='Book-Rating').fillna(0)

book_indice = CF_data1.set_index('Title-Author').index
book_vectorizer = TfidfVectorizer()
book_vectors = book_vectorizer.fit_transform(book_indice)
similarity_cf = cosine_similarity(book_vectors)


def get_title_author_from_title(df, title):
    matches = df[df['Book-Title'].str.strip().str.lower() == title.strip().lower()]
    if not matches.empty:
        return matches.iloc[0]['Title-Author']
    return None


def recommend_cbf(user_id):
    user_preferences = CBF_data.groupby('User-ID')['Title-Author'].apply(list)

    if user_id not in user_preferences:
        return {"error": f"User {user_id} not found in dataset."}

    all_titles = CBF_data['Title-Author'].unique()
    title_vectorizer = TfidfVectorizer()
    title_vectors = title_vectorizer.fit_transform(all_titles)
    similarity = cosine_similarity(title_vectors)

    book_indices = {title: index for index, title in enumerate(all_titles)}

    user_recommendations = {}
    for user, books in user_preferences.items():
        recommended_books = set()
        book_score = {}

        for book in books:
            if book not in book_indices:
                continue

            book_index = book_indices[book]
            similar_books = similarity[book_index].argsort()[::-1]
            for index in similar_books:
                similar_book = all_titles[index]
                similarity_score = similarity[book_index][index]

                if similar_book not in books:
                    recommended_books.add(similar_book)
                    book_score[similar_book] = similarity_score

                if len(recommended_books) >= 5:
                    break
            if len(recommended_books) >= 5:
                break

        user_recommendations[user] = list(recommended_books), book_score

    recommendations_list = []
    for user, (books, scores) in user_recommendations.items():
        for book in books:
            recommendations_list.append({
                'User-ID': user,
                'Title-Author': book,
                'CBF_Score': scores[book]
            })

    recommendations_df = pd.DataFrame(recommendations_list)

    CBF_unique = CBF_data.drop_duplicates(subset='Title-Author')

    recommendations_df = recommendations_df.merge(CBF_unique, on='Title-Author')
    recommendations_df = recommendations_df.rename(columns={'User-ID_x': 'User-ID'})
    recommendations_df = recommendations_df.drop(columns=['User-ID_y'])
    recommendations_cbf = recommendations_df.groupby('User-ID').agg({
        'Book-Title': list,
        'Book-Author': list,
        'Book-Rating': list,
        'num_ratings': list,
        'avg_ratings': list,
        'Image-URL-S': list,
        'Title-Author': list,
        'CBF_Score': list
    }).reset_index()

    result = recommendations_cbf[recommendations_cbf['User-ID'] == user_id]
    if result.empty:
        return {"error": "No recommendations found."}
    result['Book-Title'] = result['Book-Title'].apply(lambda x: [i if isinstance(i, str) else i[0] for i in x])
    return result.to_dict(orient='records')


def recommend_cf(user_id, book_title):
    book_name = get_title_author_from_title(CF_data1, book_title)

    if not book_name:
        return {"error": f"Book title '{book_title}' not found in dataset."}
    if book_name not in book_indice:
        return {"error": f"Book '{book_name}' not found in expert dataset."}

    if user_id not in active_user_matrix.index:
        return {"error": f"User '{user_id}' not found in active user dataset."}

    common_books = exp_user_matrix.columns.intersection(active_user_matrix.columns)
    target_user_vector = active_user_matrix.loc[user_id, common_books].values.reshape(1, -1)
    exp_user_matrix_aligned = exp_user_matrix[common_books]
    target_exp_similarities = cosine_similarity(target_user_vector, exp_user_matrix_aligned)[0]
    exp_user_similarity_dict = dict(zip(exp_user_matrix.index, target_exp_similarities))
    similar_exp_users = sorted(exp_user_similarity_dict, key=exp_user_similarity_dict.get, reverse=True)[:5]

    user_books = set(active_user_matrix.loc[user_id][active_user_matrix.loc[user_id] > 0].index)
    recommendations = []

    for exp_user in similar_exp_users:
        exp_user_books = set(exp_user_matrix.loc[exp_user][exp_user_matrix.loc[exp_user] > 0].index)
        new_recommendations = exp_user_books - user_books
        recommendations.extend(new_recommendations)

    recommendations = list(set(recommendations))

    recommendation_scores = {
        book: similarity_cf[book_indice.get_loc(book)].sum() if book in book_indice else 0
        for book in recommendations
    }

    sorted_recommendations = sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)
    sorted_recommendations = [rec for rec in sorted_recommendations if rec[0] != book_name]

    recommended_books_df = CF_data1[
        CF_data1['Title-Author'].isin([book for book, score in sorted_recommendations])].reset_index(drop=True)
    recommended_books_df['CF_Score'] = recommended_books_df['Title-Author'].map(recommendation_scores)

    return recommended_books_df.to_dict(orient='records')


def hybrid_recommend(user_id, book_name):
    cbf_recs = recommend_cbf(user_id)
    cf_recs = recommend_cf(user_id, book_name)

    if isinstance(cbf_recs, dict) and 'error' in cbf_recs:
        cbf_df = pd.DataFrame(columns=['Book-Title', 'CBF_Score'])
    else:
        cbf_df = pd.DataFrame(cbf_recs)[['Book-Title', 'CBF_Score']]

    if isinstance(cf_recs, dict) and 'error' in cf_recs:
        cf_df = pd.DataFrame(columns=['Book-Title', 'CF_Score'])
    else:
        cf_df = pd.DataFrame(cf_recs)[['Book-Title', 'CF_Score']]

    cbf_df['Book-Title'] = cbf_df['Book-Title'].apply(lambda x: x if isinstance(x, str) else x[0])
    cf_df['Book-Title'] = cf_df['Book-Title'].apply(lambda x: x if isinstance(x, str) else x[0])

    if cbf_df.empty and cf_df.empty:
        return {"error": "No recommendations available from either CBF or CF."}

    if not cbf_df.empty and 'Book-Title' in cbf_df.columns:
        cbf_df = cbf_df[['Book-Title', 'CBF_Score']]
    if not cf_df.empty and 'Book-Title' in cf_df.columns:
        cf_df = cf_df[['Book-Title', 'CF_Score']]

    merged_df = pd.merge(cbf_df, cf_df, on='Book-Title', how='outer').fillna(0)

    if merged_df.empty:
        return {"error": "No common books to merge for hybrid recommendation."}

    merged_df['CBF_Score'] = merged_df['CBF_Score'].apply(lambda x: x[0] if isinstance(x, list) else x)
    merged_df['CF_Score'] = merged_df['CF_Score'].apply(lambda x: x[0] if isinstance(x, list) else x)

    scaler = MinMaxScaler()
    merged_df[['CBF_Score', 'CF_Score']] = scaler.fit_transform(merged_df[['CBF_Score', 'CF_Score']])
    merged_df['Hybrid_Score'] = merged_df['CBF_Score'] + merged_df['CF_Score']
    merged_df = merged_df[merged_df['Book-Title'] != book_name]
    merged_df = merged_df.drop_duplicates(subset='Book-Title')
    merged_df = merged_df.sort_values(by='Hybrid_Score', ascending=False).head(5)

    # Join with CBF_data to get full book info
    top_titles = merged_df['Book-Title'].tolist()
    top_books_info = CBF_data[CBF_data['Book-Title'].isin(top_titles)].drop_duplicates(subset='Book-Title')
    final_recommendations = top_books_info.merge(merged_df, on='Book-Title', how='left')
    final_recommendations = final_recommendations.sort_values(by='Hybrid_Score', ascending=False)

    return final_recommendations.to_dict(orient='records')


def api_recommend_cbf(user_id: int = Query(..., description="User ID for CBF recommendation")):
    recs = recommend_cbf(user_id)
    if "error" in recs:
        raise HTTPException(status_code=404, detail=recs["error"])
    return {"user_id": user_id, "recommendations": recs}


def api_recommend_cf(user_id: int = Query(..., description="User ID for CF recommendation"),
                     book_name: str = Query(..., description="Book name (Title-Author) for CF recommendation")):
    recs = recommend_cf(user_id, book_name)
    if "error" in recs:
        raise HTTPException(status_code=404, detail=recs["error"])
    return {"user_id": user_id, "book_name": book_name, "recommendations": recs}


def api_recommend_hybrid(user_id: int = Query(..., description="User ID for hybrid recommendation"),
                         book_name: str = Query(..., description="Book name (Title-Author) for hybrid recommendation")):
    recs = hybrid_recommend(user_id, book_name)
    if "error" in recs:
        raise HTTPException(status_code=404, detail=recs["error"])
    return {"user_id": user_id, "book_name": book_name, "recommendations": recs}
