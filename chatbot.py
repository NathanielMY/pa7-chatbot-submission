# PA7, CS124, Stanford
# v.1.1.0
#
# Original Python code by Ignacio Cases (@cases)
# Update: 2024-01: Added the ability to run the chatbot as LLM interface (@mryan0)
# Update: 2025-01 for Winter 2025 (Xuheng Cai)
######################################################################
import util
from pydantic import BaseModel, Field

import numpy as np


# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 7."""

    def __init__(self, llm_enabled=False):
        # The chatbot's default name is `moviebot`.
        # TODO: Give your chatbot a new name.
        self.name = 'moviezmahn'

        self.llm_enabled = llm_enabled

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = util.load_ratings('data/ratings.txt')
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')

        ########################################################################
        # TODO: Binarize the movie ratings matrix.                             #
        ########################################################################

        # Binarize the movie ratings before storing the binarized matrix.
        self.ratings = self.binarize(ratings)
        self.users_ratings = []
        self.recommendations = []
        self.recommendation_idx = 0
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    ############################################################################
    # 1. WARM UP REPL                                                          #
    ############################################################################

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        ########################################################################
        # TODO: Write a short greeting message                                 #
        ########################################################################

        greeting_message = "ello boss, you want help today?"

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return greeting_message

    def goodbye(self):
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """
        ########################################################################
        # TODO: Write a short farewell message                                 #
        ########################################################################

        goodbye_message = "farewell boss, see u next time!"

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return goodbye_message
    

    ############################################################################
    # 2. Modules 2 and 3: extraction and transformation                        #
    ############################################################################

    def process(self, line):
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user
        input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this class.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'

        :param line: a user-supplied line of text
        :returns: a string containing the chatbot's response to the user input
        """
        ########################################################################
        # TODO: Implement the extraction and transformation in this method,    #
        # possibly calling other functions. Although your code is not graded   #
        # directly based on how modular it is, we highly recommended writing   #
        # code in a modular fashion to make it easier to improve and debug.    #
        ########################################################################

        if self.llm_enabled:
            response = "I processed {} in LLM Programming mode!!".format(line)
        else:
            #if user just sends yes, recommend another movie
            if line == "yes" and self.recommendation_idx > 0:
                self.recommendation_idx += 1
                response = "Based on your ratings, I recommend you watch the following movie: {}. Do you want me to recommend another movie?".format(self.recommendations[self.recommendation_idx])
            
            # Process the input to extract movie titles
            preprocessed_input = self.preprocess(line)
            titles = self.extract_titles(preprocessed_input)

            if np.count_nonzero(titles) > 5:
                #give recommendation
                if self.recommendation_idx == 0:
                    self.recommendations = self.recommend(self.users_ratings, self.ratings)
                    response = "Based on your ratings, I recommend you watch the following movie: {}. Do you want me to recommend another movie?".format(self.recommendations[self.recommendation_idx])
                    self.recommendation_idx += 1
                else:
                    response = "Based on your ratings, I recommend you watch the following movie: {}. Do you want me to recommend another movie?".format(self.recommendations[self.recommendation_idx])

            else:
                if not titles:
                    responses = [
                        "I couldn't find any movie titles in your input. Can you tell me about a movie you've watched?",
                        "Hmm, I didn't catch any movie titles there. Would you mind sharing your thoughts on a specific film?",
                        "I'm looking for movie titles in what you said, but couldn't find any. Could you mention a movie you've seen recently?"
                    ]
                    response = responses[len(preprocessed_input) % 3]
                else:
                    # Extract sentiment for the movie
                    sentiment = self.extract_sentiment(preprocessed_input)
                    
                    if sentiment > 0:
                        responses = [
                            "You liked \"{}\". Thank you for sharing! Would you like to tell me about another movie?".format(titles[0]),
                            "I see that you enjoyed \"{}\". That's great to hear! What other movies have you watched?".format(titles[0]),
                            "Ah, so you're a fan of \"{}\". I'll remember that. Any other films you'd like to discuss?".format(titles[0])
                        ]
                        response = responses[len(preprocessed_input) % 3]

                        for title in titles:
                            self.users_ratings[title] = 1

                    
                    elif sentiment < 0:
                        responses = [
                            "You didn't like \"{}\". I'll make a note of that. Tell me about another movie you've watched.".format(titles[0]),
                            "I understand that \"{}\" wasn't your cup of tea. What's another movie you've seen?".format(titles[0]),
                            "Sorry to hear you didn't enjoy \"{}\". Perhaps you could share your thoughts on a different film?".format(titles[0])
                        ]
                        response = responses[len(preprocessed_input) % 3]

                        for title in titles:
                            self.users_ratings[title] = -1
                    else:
                        responses = [
                            "I'm not sure if you liked \"{}\". Can you tell me more about it?".format(titles[0]),
                            "Your feelings about \"{}\" aren't clear to me. Could you elaborate on what you thought of it?".format(titles[0]),
                            "I'm having trouble determining your opinion of \"{}\". Would you mind clarifying whether you enjoyed it?".format(titles[0])
                        ]
                        response = responses[len(preprocessed_input) % 3]

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return response

    @staticmethod
    def preprocess(text):
        """Do any general-purpose pre-processing before extracting information
        from a line of text.

        Given an input line of text, this method should do any general
        pre-processing and return the pre-processed string. The outputs of this
        method will be used as inputs (instead of the original raw text) for the
        extract_titles, extract_sentiment, extract_sentiment_for_movies, and
        extract_emotion methods.

        Note that this method is intentially made static, as you shouldn't need
        to use any attributes of Chatbot in this method.

        :param text: a user-supplied line of text
        :returns: the same text, pre-processed
        """
        ########################################################################
        # TODO: Preprocess the text into a desired format.                     #
        # NOTE: This method is completely OPTIONAL. If it is not helpful to    #
        # your implementation to do any generic preprocessing, feel free to    #
        # leave this method unmodified.                                        #
        ########################################################################

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return text

    def extract_titles(self, preprocessed_input):
        """Extract potential movie titles from a line of pre-processed text.

        Given an input text which has been pre-processed with preprocess(),
        this method should return a list of movie titles that are potentially
        in the text.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: list of movie titles that are potentially in the text
        """
        titles = []
        # Look for text between quotation marks
        in_quotes = False
        current_title = ""
        
        for char in preprocessed_input:
            if char == '"':
                if in_quotes:
                    # End of a title
                    if current_title:  # Only add non-empty titles
                        titles.append(current_title)
                    current_title = ""
                in_quotes = not in_quotes
            elif in_quotes:
                # Add character to current title
                current_title += char
                
        return titles

    def find_movies_by_title(self, title):
        """ Given a movie title, return a list of indices of matching movies.

        - If no movies are found that match the given title, return an empty
        list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a
        list
        that contains the index of that matching movie.

        Example:
          ids = chatbot.find_movies_by_title('Titanic')
          print(ids) // prints [1359, 2716]

        :param title: a string containing a movie title
        :returns: a list of indices of matching movies
        """
        
    def find_movies_by_title(self, title):
        file_path = "./data/movies.txt"  # Adjust if needed
        matching_indices = []

        # Check if the title contains a year at the end
        if "(" in title and ")" in title and title.strip()[-1] == ")":
            parts = title.rsplit(" (", 1)
            search_title = parts[0].strip()
            search_year = parts[1][:-1]  # Remove closing ')'
        else:
            search_title = title.strip()
            search_year = None

        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split('%')
                if len(parts) < 2:
                    continue  # Skip malformed lines
                
                index, movie_entry = parts[0], parts[1]

                # Extract title and year manually
                if "(" in movie_entry and ")" in movie_entry and movie_entry.strip()[-1] == ")":
                    movie_parts = movie_entry.rsplit(" (", 1)
                    movie_title = movie_parts[0].strip()
                    movie_year = movie_parts[1][:-1]  # Remove closing ')'
                else:
                    movie_title = movie_entry.strip()
                    movie_year = None

                # Match based on title and year (if provided)
                if search_title.lower() == movie_title.lower() and (search_year is None or search_year == movie_year):
                    matching_indices.append(int(index))

        return matching_indices


    def extract_sentiment(self, preprocessed_input):
        """Extract a sentiment rating from a line of pre-processed text.

        You should return -1 if the sentiment of the text is negative, 0 if the
        sentiment of the text is neutral (no sentiment detected), or +1 if the
        sentiment of the text is positive.

        Example:
          sentiment = chatbot.extract_sentiment(chatbot.preprocess(
                                                    'I liked "The Titanic"'))
          print(sentiment) // prints 1

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a numerical value for the sentiment of the text
        """
        # Remove movie titles from input to avoid counting them in sentiment analysis
        titles = self.extract_titles(preprocessed_input)
        for title in titles:
            preprocessed_input = preprocessed_input.replace(f'"{title}"', '')
        
        # Convert input to lowercase and split into words
        words = preprocessed_input.lower().split()
        
        # Create bag of words (word -> count)
        bag_of_words = {}
        for word in words:
            # Remove punctuation from word
            word = ''.join(c for c in word if c.isalpha())
            if word:  # Skip empty strings
                bag_of_words[word] = bag_of_words.get(word, 0) + 1
        
        net_word_sentiment = 0
        for word, count in bag_of_words.items():
            net_word_sentiment += count * self.calc_word_sentiment(word)
        
        return (net_word_sentiment > 0) - (net_word_sentiment < 0)

    
    def calc_word_sentiment(self, word):
       #returns 1 if positive, -1 if negative, 0 if neutral
       with open('./data/sentiment.txt', 'r') as f:
           sentiment_lines = f.readlines()
       
       # Process each line to create a dictionary of word sentiments
       for line in sentiment_lines:
           # Remove whitespace and split by comma
           parts = line.strip().split(',')
           if len(parts) == 2:
               sentiment_word, sentiment_value = parts
               # Check if this is the word we're looking for
               if sentiment_word == word:
                   # Return sentiment value based on pos/neg
                   if sentiment_value == 'pos':
                       return 1
                   elif sentiment_value == 'neg':
                       return -1
                   else:
                       return 0
       
       # If word not found in sentiment dictionary, return neutral sentiment
       return 0


    ############################################################################
    # 3. Movie Recommendation helper functions                                 #
    ############################################################################

    @staticmethod
    def binarize(ratings, threshold=2.5):
        """Return a binarized version of the given matrix.

        To binarize a matrix, replace all entries above the threshold with 1.
        and replace all entries at or below the threshold with a -1.

        Entries whose values are 0 represent null values and should remain at 0.

        Note that this method is intentionally made static, as you shouldn't use
        any attributes of Chatbot like self.ratings in this method.

        :param ratings: a (num_movies x num_users) matrix of user ratings, from
         0.5 to 5.0
        :param threshold: Numerical rating above which ratings are considered
        positive

        :returns: a binarized version of the movie-rating matrix
        """
        ########################################################################
        # TODO: Binarize the supplied ratings matrix.                          #
        #                                                                      #
        # WARNING: Do not use self.ratings directly in this function.          #
        ########################################################################

        # The starter code returns a new matrix shaped like ratings but full of
        # zeros.
        binarized_ratings = np.where(ratings > threshold, 1, -1)
        binarized_ratings[ratings == 0] = 0
        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return binarized_ratings

    def similarity(self, u, v):
        """Calculate the cosine similarity between two vectors.

        You may assume that the two arguments have the same shape.

        :param u: one vector, as a 1D numpy array
        :param v: another vector, as a 1D numpy array

        :returns: the cosine similarity between the two vectors
        """
        ########################################################################
        # TODO: Compute cosine similarity between the two vectors.             #
        ########################################################################
        similarity = 0

        u_norm = np.linalg.norm(u)
        v_norm = np.linalg.norm(v)
        
        if u_norm == 0 or v_norm == 0:
            return 0
        
        # Calculate dot product and divide by the product of magnitudes
        similarity = np.dot(u, v) / (u_norm * v_norm)
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return similarity

    def recommend(self, user_ratings, ratings_matrix, k=10, llm_enabled=False):
        """Generate a list of indices of movies to recommend using collaborative
         filtering.

        You should return a collection of `k` indices of movies recommendations.

        As a precondition, user_ratings and ratings_matrix are both binarized.

        Remember to exclude movies the user has already rated!

        Please do not use self.ratings directly in this method.

        :param user_ratings: a binarized 1D numpy array of the user's movie
            ratings
        :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
          `ratings_matrix[i, j]` is the rating for movie i by user j
        :param k: the number of recommendations to generate
        :param llm_enabled: whether the chatbot is in llm programming mode

        :returns: a list of k movie indices corresponding to movies in
        ratings_matrix, in descending order of recommendation.
        """

        ########################################################################
        # TODO: Implement a recommendation function that takes a vector        #
        # user_ratings and matrix ratings_matrix and outputs a list of movies  #
        # recommended by the chatbot.                                          #
        #                                                                      #
        # WARNING: Do not use the self.ratings matrix directly in this         #
        # function.                                                            #
        #                                                                      #
        # For GUS mode, you should use item-item collaborative filtering with  #
        # cosine similarity, no mean-centering, and no normalization of        #
        # scores.                                                              #
        ########################################################################

        # Populate this list with k movie indices to recommend to the user.

        recommendations = []
        #create a nxn matrix of similarity scores of movies with ach other
        temp_matrix = np.zeros((ratings_matrix.shape[0], ratings_matrix.shape[0]))

        for i in range(ratings_matrix.shape[0]):
            for j in range(ratings_matrix.shape[0]):
                temp_matrix[i][j] = self.similarity(ratings_matrix[i], ratings_matrix[j])

        #binarize user ratinfs
        bin_user_ratings = self.binarize(user_ratings)

        movie_idx_to_ratr = {}

        
        for j in range(ratings_matrix.shape[0]):
            aggr = 0
            for i in range(len(bin_user_ratings)):
                if bin_user_ratings[i] != 0:
                    sim_score = temp_matrix[j][i]
                    aggr += sim_score * bin_user_ratings[i]
            movie_idx_to_ratr[j] = aggr
        
        #return top k keys with highest values
        recommendations = sorted(movie_idx_to_ratr, key=movie_idx_to_ratr.get, reverse=True)[:k]
                

        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return recommendations

    ############################################################################
    # 4. PART 2: LLM Prompting Mode                                            #
    ############################################################################

    def llm_system_prompt(self):
        """
        Return the system prompt used to guide the LLM chatbot conversation.

        NOTE: This is only for LLM Mode!  In LLM Programming mode you will define
        the system prompt for each individual call to the LLM.
        """
        ########################################################################
        # TODO: Write a system prompt message for the LLM chatbot              #
        ########################################################################

        system_prompt = """Your name is moviebot. You are a movie recommender chatbot. """ +\
        """You can help users find movies they like and provide information about movies."""

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################

        return system_prompt
    
    ############################################################################
    # 5. PART 3: LLM Programming Mode (also need to modify functions above!)   #
    ############################################################################

    def extract_emotion(self, preprocessed_input):
        """LLM PROGRAMMING MODE: Extract an emotion from a line of pre-processed text.
        
        Given an input text which has been pre-processed with preprocess(),
        this method should return a list representing the emotion in the text.
        
        We use the following emotions for simplicity:
        Anger, Disgust, Fear, Happiness, Sadness and Surprise
        based on early emotion research from Paul Ekman.  Note that Ekman's
        research was focused on facial expressions, but the simple emotion
        categories are useful for our purposes.

        Example Inputs:
            Input: "Your recommendations are making me so frustrated!"
            Output: ["Anger"]

            Input: "Wow! That was not a recommendation I expected!"
            Output: ["Surprise"]

            Input: "Ugh that movie was so gruesome!  Stop making stupid recommendations!"
            Output: ["Disgust", "Anger"]

        Example Usage:
            emotion = chatbot.extract_emotion(chatbot.preprocess(
                "Your recommendations are making me so frustrated!"))
            print(emotion) # prints ["Anger"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()

        :returns: a list of emotions in the text or an empty list if no emotions found.
        Possible emotions are: "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"
        """
        return []

    ############################################################################
    # 6. Debug info                                                            #
    ############################################################################

    def debug(self, line):
        """
        Return debug information as a string for the line string from the REPL

        NOTE: Pass the debug information that you may think is important for
        your evaluators.
        """
        debug_info = 'debug info'
        return debug_info

    ############################################################################
    # 7. Write a description for your chatbot here!                            #
    ############################################################################
    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your
        chatbot can do and how the user can interact with it.

        NOTE: This string will not be shown to the LLM in llm mode, this is just for the user
        """

        return """
        Your task is to implement the chatbot as detailed in the PA7
        instructions.
        Remember: in the GUS mode, movie names will come in quotation marks
        and expressions of sentiment will be simple!
        TODO: Write here the description for your own chatbot!
        """


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    print('    python3 repl.py')
