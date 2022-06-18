class LabelTransformations:

    def __call__(self, labels):
        transformed_labels = []
        for label in labels:
            transformed_labels += list(self.__transform__(label))

        return transformed_labels

    def __transform__(self, label):
        label_words = label.split(' ')

        '''create camelcase label'''
        camelcase_label = self.__camelcase_join__(label_words)

        '''create all_words_capital joined with hyphen, underscore and space'''
        all_words_capital_label = self.__all_words_capital__(label_words)

        all_words_capital_label_hyphen_joined = self.__hyphen_join__(all_words_capital_label)
        all_words_capital_label_underscore_joined = self.__underscore_join__(all_words_capital_label)
        all_words_capital_label_space_joined = self.__space_join__(all_words_capital_label)
        all_words_capital_label_joined = self.__join__(all_words_capital_label)

        '''create all_words_start_capital joined with hyphen, underscore and space'''
        all_words_start_capital_label = self.__all_words_start_capital__(label_words)

        all_words_start_capital_label_hyphen_joined = self.__hyphen_join__(all_words_start_capital_label)
        all_words_start_capital_label_underscore_joined = self.__underscore_join__(all_words_start_capital_label)
        all_words_start_capital_label_space_joined = self.__space_join__(all_words_start_capital_label)
        all_words_start_capital_label_joined = self.__join__(all_words_start_capital_label)

        '''create all_words_start_capital joined with hyphen, underscore and space'''
        all_words_small_label = self.__all_words_small__(label_words)

        all_words_small_label_hyphen_joined = self.__hyphen_join__(all_words_small_label)
        all_words_small_label_underscore_joined = self.__underscore_join__(all_words_small_label)
        all_words_small_label_space_joined = self.__space_join__(all_words_small_label)
        all_words_small_label_joined = self.__join__(all_words_small_label)

        return camelcase_label, \
                all_words_capital_label_hyphen_joined, all_words_capital_label_underscore_joined, all_words_capital_label_space_joined, all_words_capital_label_joined,\
                all_words_start_capital_label_hyphen_joined, all_words_start_capital_label_underscore_joined, all_words_start_capital_label_space_joined, all_words_start_capital_label_joined, \
                all_words_small_label_hyphen_joined, all_words_small_label_underscore_joined, all_words_small_label_space_joined, all_words_small_label_joined

    def __underscore_join__(self, label_list):
        return '_'.join(label_list)

    def __hyphen_join__(self, label_list):
        return '-'.join(label_list)

    def __space_join__(self, label_list):
        return ' '.join(label_list)
    
    def __join__(self, label_list):
        return ''.join(label_list)

    def __camelcase_join__(self, label_list):
        start_word = label_list[0]
        following_words = label_list[1:]
        following_words = self.__all_words_start_capital__(following_words)
        return ''.join([start_word.lower()] + following_words)

    def __all_words_capital__(self, label_list):
        return [word.upper() for word in label_list]

    def __all_words_start_capital__(self, label_list):
        #label = ' '.join(label_list)
        return [x.title() for x in label_list if not x.isspace()]

    def __all_words_small__(self, label_list):
        return [word.lower() for word in label_list]
        

class DataTransformations:
    def __seperate_using_hyphens__(self, data, indices):
        for i in reversed(indices):
            data = data[:i] + '-' + data[i:]
        return data

    def __seperate_using_spaces__(self, data, indices):
        for i in reversed(indices):
            data = data[:i] + ' ' + data[i:]
        return data


