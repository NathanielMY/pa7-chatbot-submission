# PA7, CS124, Stanford
# v.1.1.0
#
# Original Python code by Ignacio Cases (@cases)
# Update: 2024-01: Added the ability to run the chatbot as LLM interface (@mryan0)
# Update: 2025-01 for Winter 2025 (Xuheng Cai)
######################################################################
import util
from pydantic import BaseModel, Field
from porter_stemmer import PorterStemmer

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
        self.users_ratings = [0 for _ in range(len(self.ratings))]
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
            yes_words = ["yes", "y", "yeah", "yup", "yep", "sure", "ok", "okay", "alright", "alrighty", "alrighty-then", "certainly", "definitely", "absolutely", "for sure", "for real"]
            no_words = ["no", "n", "nah", "nope", "not", "no way", "no thanks", "no thanks", "no way", "no way jose", "no way josé", "no way jose", "no way josé", "no way jose", "no way josé", "no way jose", "no way josé"]
            # Strip line from punctuation and check if the cleaned text is in yes_words
            cleaned_line = line.lower().strip('!?.;:')
            if cleaned_line in yes_words and self.recommendation_idx > 0:
                self.recommendation_idx += 1
                response = "Based on your ratings, I recommend you watch the following movie: {}. Do you want me to recommend another movie?".format(self.titles[self.recommendations[self.recommendation_idx]][0])
                return response
            if cleaned_line in no_words and self.recommendation_idx > 0:
                return "ok I won't recommend another movie. Let me know if you have any other movie recommendations."
            
            # Process the input to extract movie titles
            preprocessed_input = self.preprocess(line)
            movie_titles = self.extract_titles(preprocessed_input)
            #call get movie titles on each index
            movie_indices = []
            for title in movie_titles:
                movie_indices.extend(self.find_movies_by_title(title))

            # Concatenate multiple movie titles into a single string
            if len(movie_titles) > 1:
                if len(movie_titles) == 2:
                    # For two movies, use "and" to join them
                    concatenated_titles = f'"{movie_titles[0]}" and "{movie_titles[1]}"'
                else:
                    # For more than two movies, use commas and "and" for the last one
                    titles_with_quotes = [f'"{title}"' for title in movie_titles]
                    last_title = titles_with_quotes.pop()
                    concatenated_titles = ", ".join(titles_with_quotes) + ", and " + last_title

                # Replace individual titles with the concatenated version in later code
                # Store the original titles for processing individual sentiments
                original_titles = movie_titles.copy()
                movie_titles = [concatenated_titles]

            
            
            if not movie_indices:
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
                        "You liked \"{}\". Thank you for sharing! Would you like to tell me about another movie?".format(movie_titles[0]),
                        "I see that you enjoyed \"{}\". That's great to hear! What other movies have you watched?".format(movie_titles[0]),
                        "Ah, so you're a fan of \"{}\". I'll remember that. Any other films you'd like to discuss?".format(movie_titles[0])
                    ]
                    response = responses[len(preprocessed_input) % 3]

                    for index in movie_indices:
                        self.users_ratings[index] = 1

                
                elif sentiment < 0:
                    responses = [
                        "You didn't like \"{}\". I'll make a note of that. Tell me about another movie you've watched.".format(movie_titles[0]),
                        "I understand that \"{}\" wasn't your cup of tea. What's another movie you've seen?".format(movie_titles[0]),
                        "Sorry to hear you didn't enjoy \"{}\". Perhaps you could share your thoughts on a different film?".format(movie_titles[0])
                    ]
                    response = responses[len(preprocessed_input) % 3]

                    for index in movie_indices:
                        self.users_ratings[index] = -1
                else:
                    responses = [
                        "I'm not sure if you liked \"{}\". Can you tell me more about it?".format(movie_titles[0]),
                        "Your feelings about \"{}\" aren't clear to me. Could you elaborate on what you thought of it?".format(movie_titles[0]),
                        "I'm having trouble determining your opinion of \"{}\". Would you mind clarifying whether you enjoyed it?".format(movie_titles[0])
                    ]
                    response = responses[len(preprocessed_input) % 3]

                if np.count_nonzero(self.users_ratings) >= 5:
                        #give recommendation
                        if self.recommendation_idx == 0:
                            self.recommendations = self.recommend(self.users_ratings, self.ratings)
                            #this is just an index, now need to get the title
                            recommended_movie_titles = [self.titles[i] for i in self.recommendations]
                            response = "Based on your ratings, I recommend you watch the following movie: {}. Do you want me to recommend another movie?".format(recommended_movie_titles[self.recommendation_idx][0])
                            self.recommendation_idx += 1
                        else:
                            response = "Based on your ratings, I recommend you watch the following movie: {}. Do you want me to recommend another movie?".format(recommended_movie_titles[self.recommendation_idx][0])


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

    def normalize_title(self,title):
        title = title.replace(",", "").replace(".", "").replace("'", "").replace(":", "").replace(";", "").replace("-", "")

        words = title.lower().split()

 
        if words and words[0] in {"the", "a", "an"}:
            words.append(words.pop(0))  

        return set(words)

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
        
        movie_file = "./data/movies.txt"
        matching_indices = []
        search_words = self.normalize_title(title)

        has_year = False
        for word in search_words:
            if word.startswith('(') and word.endswith(')') and all(char.isdigit() for char in word[1:-1]):
                has_year = True
                break
        
        with open(movie_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split('%')

                index, movie_entry = parts[0].strip(), parts[1].strip()
                movie_words = self.normalize_title(movie_entry)
     
                if not has_year:
                    #remove the year from the movie_words
                    movie_words = [word for word in movie_words if not (word.startswith('(') and word.endswith(')') and all(char.isdigit() for char in word[1:-1]))]
                    movie_words = set(movie_words)
                    #print(movie_words)
                
                if search_words.issubset(movie_words) and movie_words.issubset(search_words):
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

        stemmer = PorterStemmer()

        # Track negation words
        negation_words = {"no", "not", "never", "didn't", "don't", "doesn't", "can't", "cannot", "wasn't", "weren't", "haven't", "hasn't", "won't", "wouldn't", "couldn't", "shouldn't"}
        
        # Track intensifiers
        intensifiers = {"really", "very", "extremely", "so", "totally", "absolutely", "completely"}
        
        net_sentiment = 0
        negation = False
        intensifier = 1
        
        for i, word in enumerate(words):
            # Clean the word of punctuation
            
            clean_word = stemmer.stem(word)
            #edge case with enjoy for the stemmer
            if clean_word == "enjoi":
                clean_word = "enjoy"

            if not clean_word:
                continue
                
            # Check for negation
            if clean_word in negation_words:
                negation = not negation  # Toggle negation
                continue
                
            # Check for intensifiers
            if clean_word in intensifiers:
                intensifier = 2
                continue


            
            word_sentiment = self.calc_word_sentiment(clean_word)


            #print(clean_word, word_sentiment, negation, intensifier)
            
            # Apply negation and intensifier
            if word_sentiment != 0:
                if negation:
                    word_sentiment = -word_sentiment
                
                net_sentiment += word_sentiment * intensifier
                intensifier = 1  # Reset intensifier after applying
            
            # Reset negation after 3 words if not used
            # if negation and i > 0 and i % 3 == 0:
            #     negation = False
        
        # Return final sentiment
        if net_sentiment > 0:
            return 1
        elif net_sentiment < 0:
            return -1
        else:
            return 0

    
    def calc_word_sentiment(self, word):
        """Returns 1 if positive, -1 if negative, 0 if neutral"""
        # Use a class attribute to store the sentiment dictionary
        if not hasattr(self, 'sentiment_dict'):
            self.sentiment_dict = {}
            with open('./data/sentiment.txt', 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and ',' in line:  # Skip empty lines
                        parts = line.split(',')
                        if len(parts) == 2:
                            sentiment_word, sentiment_value = parts
                            # Store sentiment value in dictionary
                            if sentiment_value == 'pos':
                                self.sentiment_dict[sentiment_word] = 1
                            elif sentiment_value == 'neg':
                                self.sentiment_dict[sentiment_word] = -1
        
        # Look up the word in our dictionary
        return self.sentiment_dict.get(word, 0)  # Return 0 if word not found


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
        """Generate a list of indices of movies to recommend using collaborative filtering."""
        
        # Dictionary to store predicted ratings for unwatched movies
        movie_idx_to_rating = {}
        
        # For each movie the user hasn't rated
        for j in range(ratings_matrix.shape[0]):
            if user_ratings[j] != 0:  # Skip movies the user has already rated
                continue
            
            # Calculate predicted rating for movie j
            numerator = 0
            denominator = 0
            
            # Compare with all movies the user has rated
            for i in range(ratings_matrix.shape[0]):
                if user_ratings[i] != 0:  # Only consider movies the user has rated
                    # Calculate similarity between movie i and movie j
                    sim_score = self.similarity(ratings_matrix[i], ratings_matrix[j])
                    numerator += sim_score * user_ratings[i]
                    denominator += abs(sim_score)
            
            # Store predicted rating if we have valid data
            if denominator > 0:
                movie_idx_to_rating[j] = numerator
        
        # Return top k movies with highest predicted ratings
        recommendations = sorted(movie_idx_to_rating, key=movie_idx_to_rating.get, reverse=True)[:k]
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
