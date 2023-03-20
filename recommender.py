import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class BookRecommender:
    '''Book recommender class. Uses the data stored in /data to recommend similar books by comparing tags goodreads users have labelled the book with.
    
    attributes
    ----------
    tag_num: int
        The number of book tags to use for book comparisons. Only the most popular tags are used.

    book_data: np.array
        Array containing relevant information on the books in the dataset.

    tag_data: np.array
        Array containing data about which book has been labeled with which tag.

    tag_info: np.array
        Array holding tag ids and corresponding tag names.

    tag_counts: np.array
        Array counting total tag appearances.

    book_dir_vecs: dict
        Dict holding direction vectors for all books in the database.
    
    methods
    -------
    make_dir_vector(book_id : int) -> np.array
        Creates a direction vector for a book given its id.

    get_recom_matrix(self, dir_vec: np.array) -> np.array:
        returns recommendation matrix with book ids in first column, cosine similarities in the second.

    recommend(self, book_id : int, rec_num : int = 5) -> None:
        prints book recommendations given a book id.
    '''

    def __init__(self, tag_num : int = 100):
        self.tag_num = tag_num

        print("reading data...")
        self.book_data = np.loadtxt('data/books.csv', dtype=str, delimiter=',', skiprows=1, usecols=(1, 7, 10), encoding='utf-8', comments=None, quotechar='"')
        self.tag_data = np.loadtxt('data/book_tags.csv', dtype=int, delimiter=',', skiprows=1)
        self.tag_info = np.loadtxt('data/tags.csv', dtype=str, delimiter=',', skiprows=1, encoding='utf-8')

        print("preprocessing tag data...")
        tag_data_sorted = self.tag_data[np.argsort(self.tag_data[:,1])]
        self.tag_counts = np.zeros((len(np.unique(tag_data_sorted[:,1])),2), dtype=int)
        self.tag_counts[:,0] = np.unique(tag_data_sorted[:,1])
        for i in range(np.shape(tag_data_sorted)[0]):
            self.tag_counts[tag_data_sorted[i,1],1] += tag_data_sorted[i,2]
        self.tag_counts = self.tag_counts[np.argsort(self.tag_counts[:,1])[-1:-(self.tag_num+1):-1]] #has tag id in the first col, total amount in the second

        print("preprocessing book tags...")
        self.book_dir_vecs = {}
        for entry in self.book_data:
            self.book_dir_vecs[entry[0]] = self.make_dir_vector(int(entry[0]))

        print("---------------------------")

    def make_dir_vector(self, book_id: int) -> np.array:
        '''Creates a 'direciton vector' for a book given its id. The direction vector encodes information about which tags the book is labeled with.
        
        parameters
        ---------
        book_id: int
            id of the book.

        returns
        -------
        dir_vec: np.array of length self.tag_num
            direction vector encoding which tags the book has been labeled with.
        '''

        dir_vec = np.zeros(self.tag_num, dtype=int)     #the direction vector

        #find the relevant part of the book_data
        rows = np.argwhere(self.tag_data[:,0] == book_id)
        relevant = self.tag_data[rows[0][0]:(rows[-1][0]+1), :]

        #construct direction vector
        for i in range(self.tag_num):
            rel_row = np.argwhere(relevant[:,1] == self.tag_counts[i,0])
            if len(rel_row) != 0:
                dir_vec[i] += relevant[rel_row[0][0], 2]

        return dir_vec

    def get_recom_matrix(self, dir_vec: np.array) -> np.array:
        '''returns recommendation matrix with book ids in first column, cosine similarities in the second.
        
        parameters
        ----------
        dir_vec: np.array
            direction vector as returned by make_dir_vector.

        returns
        -------
        res: np.array
            recommendation matrix with book ids in first column, cosine similarities in the second, sorted by cosine similarity.
        '''

        X = dir_vec.reshape(1,-1)
        Y = [item for item in self.book_dir_vecs.values()]
        sim = cosine_similarity(X,Y)[0]
        res = np.array([[int(key), 0.] for key in self.book_dir_vecs.keys()])
        for i in range(len(sim)):
            res[i, 1] = sim[i]
        return res[np.argsort(res[:,1])[-1::-1]]
    
    def recommend(self, book_id : int, rec_num : int = 5) -> None:
        '''prints book recommendations given a book id.
        
        parameters
        ----------
        book_id: int
            id of the book for which to give recommendations.

        rec_num: int, optional
            number of recommendations to print.

        returns
        -------
        None
        '''

        try:
            curr_index = np.argwhere(self.book_data[:,0] == str(book_id))[0][0]
        except IndexError:
            raise IndexError('book id is not found in book data.')

        matrix = self.get_recom_matrix(self.book_dir_vecs[str(book_id)])

        print("Recommendations for {book_title} by {author}:".format(book_title=self.book_data[curr_index, 2], author=self.book_data[curr_index, 1]))

        for entry in matrix[1:(rec_num+1),:]:
            index = np.argwhere(self.book_data[:,0] == str(int(entry[0])))[0][0]
            print(" +++ {title} by {author}".format(title=self.book_data[index,2], author=self.book_data[index,1]))

if __name__ == '__main__':
    R = BookRecommender(tag_num=300)
    while(True):
        x = input("Please enter a valid book id for which you would like to receive recommendations. Enter 'x' to quit.\n")
        if x == 'x':
            break
        try:
            x = int(x)
        except ValueError:
            print("Please enter a valid number.")
            continue

        try:
            R.recommend(x)
        except IndexError:
            print("Book id is not found in database. Please try again.")    
