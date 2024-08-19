import learner  # Assuming learner.py contains functions for user training
import recognizer  # Assuming recognizer.py contains functions for face recognition
import dbconn  # Assuming dbconn.py handles database connections and user operations

def get_user_action():
    """Prompts the user for action and validates input."""
    print("Sign up enter 1")
    print("Sign in enter 2\n")
    print("Select action from above two.")
    while True:
        action = input('Select action (1 or 2): ')
        try:
            action = int(action)
            if action not in (1, 2):
                raise ValueError
            return action
        except ValueError:
            print("\nInvalid action. Please enter 1 or 2.")

def sign_up_user():
    """Handles user sign-up process, including name input, database creation,
       and training."""
    email = input('Enter email : ')
    try:
        if dbconn.create_user(email, name=None):  # Assume name is captured elsewhere
            id, name = dbconn.get_user(email)
            res_train = learner.learn_user(id)
            if res_train:
                print("\nUser sign up successful.")
            else:
                # Handle training failure (e.g., log error, retry)
                print("\nUser sign up unsuccessful due to training failure.")
                dbconn.del_user(id)  # Consider handling potential deletion errors
        else:
            print('\nEmail address already exists.')
    except Exception as e:
        print(f"\nAn error occurred during sign-up: {e}")

def sign_in_user():
    """Handles user sign-in process, retrieving user data and initiating recognition."""
    email = input('Enter email : ')
    res = dbconn.get_user(email)
    if res:
        id, name = res
        recognizer.recognize_face(id, name)  # Call recognition function
    else:
        print('\nPlease sign up first.')

def main():
    """Manages the overall program flow and error handling."""
    while True:
        try:
            action = get_user_action()
            if action == 1:
                sign_up_user()
            elif action == 2:
                sign_in_user()
            else:
                print("Unexpected action. Exiting.")
                break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")

if __name__ == '__main__':
    main()

