import numpy as np
import torch

class CobotEnv:
    """
    The Cobot environment
    """
    #constants
    ROBOT_PROCESS_TIME = [0.372,1.1,0.685,0.47,0.489,0.271,1.1,0.62,0.333,0.23,0.878,0.809,0.711] # taken from data, constant
    SLOW_OPERATOR_EXECUTION_TIME = [0.5,0.667,0.333,1,0.5,0.5,0.333,1,0.667,0.5,0.667,0.5,1]
    EXPERT_OPERATOR_EXECUTION_TIME = [round(mu*0.8, 3) for mu in SLOW_OPERATOR_EXECUTION_TIME] # 80% of slow operator, used as mean values
    MU_OPERATORS = (SLOW_OPERATOR_EXECUTION_TIME,EXPERT_OPERATOR_EXECUTION_TIME)
    STD = 0.04
    

    def __init__(self, n_operators: int = 2, robot_execution_time: list = ROBOT_PROCESS_TIME, id_operator: int = 0, mu_operators: list = MU_OPERATORS, std: float = STD) -> None: #std: float = 0.04
        """
        Initializes an Cobot environment.

        Parameters
        ----------
            n_operators: int
                Number of human operators.
            robot_execution_time: list
                List containing the robot task processing time.
            id_operator: int
                Id of the operator.
            mu_operators: tuple
                Tuple of shape (n_operators x n_tasks) containing the mean values of the normal distribution for each task for each human operator.
            std: float
                Standard deviation of the normal distribution used to sample operators' processing time.
        """
        self.std = std
        self.robot_execution_time = np.array(robot_execution_time) # robot tasks execution time
        self.n_operators = n_operators
        self.mu_operators = mu_operators
        assert len(self.mu_operators) == self.n_operators, "Invalid Input!" # 'mu_operators' must have the same length as n_operators

        # Ids of the tasks performable by robot and human operators
        self.robot_task_id = np.array((7,8,9,11,12,13,14,15,16,17,18,19,20))
        self.operator_task_id = np.array((1,2,3,4,5,6,7,8,9,10,12,13,14))

        #set the initial state and sample new processing time of human operators
        self.reset(id_operator)


    def reset(self, id_operator: int = None) -> tuple:
        """
        Reset the environment to the initial state and re-sample new processing time of the human operators and returns the initial observation.

        Parameters
        ----------
            id_operator: int
                Id of the new operator, None otherwise.

        Returns
        -------
            tuple
                Initial state of the environment.
        """

        self.operators_sampled_time = self.sample_process_time() #operators processing time: (n_operators x n_tasks)
        
        if id_operator is not None:
            self.id_operator = id_operator
        
        # State
        self.robot_done = np.zeros(len(self.robot_task_id), dtype='int') # 0 = not done, 1 = done
        self.robot_scheduled = 0 # id of the task the robot has scheduled, 0 otherwise
        self.operator_done = np.zeros(len(self.operator_task_id), dtype='int') # 0 = not done, 1 = done
        self.operator_scheduled = 0 # id of the task the operator has scheduled, 0 otherwise
        self.operator_execution_time = np.copy(self.operators_sampled_time[self.id_operator]) # operators tasks execution time

        return self.get_state()


    def step(self, action: int, new_id_operator: int = None) -> tuple[tuple, float, bool]:
        """
        Simulates scheduling action and calculates rewards.

        Parameters
        ----------
            action: int
                Id of the next scheduled tasks.
            new_id_operator: Optional[int]
                Id of the new operator, None otherwise.
        
        Returns
        -------
            tuple, float, bool
                Current state of the environment, reward, whether the episode is over or not.
        """

        assert (action >= min(min(self.robot_task_id),min(self.operator_task_id))) and (action <= max(max(self.robot_task_id),max(self.operator_task_id))), "Invalid input! Task ID out of bound."

        robot_task_done = self.robot_task_id[self.robot_done == 1]
        operator_task_done = self.operator_task_id[self.operator_done == 1]
        task_done = np.concatenate((robot_task_done, operator_task_done))
        assert (action not in task_done), "Invalid input! Task already done!"
        
        # change operator and update execution time in state
        if new_id_operator is not None:
            self.set_operator(new_id_operator)

        initial_time = self.get_total_time()

        #######################
        ### Task assignment ###
        #######################
        
        # both robot and operator have no task scheduled
        # REMEMBER: in this case both robot and operator always have at least one schedulable task, and there are at least 2 tasks left
        if self.operator_scheduled == 0 and self.robot_scheduled == 0:

            assert action in self.robot_task_id, "Invalid input! Task not schedulable to robot!"
            
            # assign to robot
            self.robot_scheduled = action
            
            if self.check_and_finish():
                return self.get_state(), initial_time-self.get_total_time(), True

            # end step
            return self.get_state(), 0, False

        # robot has no task scheduled
        elif self.robot_scheduled == 0:

            assert action in self.robot_task_id, "Invalid input! Task not schedulable to robot!"
            assert (action != self.operator_scheduled), "Invalid input! Task already scheduled!"

            # update state - robot
            self.robot_scheduled = action

        # operator has no task scheduled
        else:

            assert action in self.operator_task_id, "Invalid input! Task not schedulable to operator!"
            assert (action != self.robot_scheduled), "Invalid input! Task already scheduled!"

            # update state - operator
            self.operator_scheduled = action
        
        ########################
        ### Task fulfillment ###
        ########################
        
        # accomplished robot tasks time + robot scheduled task ex. time
        robot_time = np.sum(self.robot_done*self.robot_execution_time) + self.robot_execution_time[np.where(self.robot_task_id==self.robot_scheduled)[0][0]]
        # accomplished operator tasks time + operator scheduled task ex. time
        operator_time = np.sum(self.operator_done*self.operator_execution_time) + self.operator_execution_time[np.where(self.operator_task_id==self.operator_scheduled)[0][0]]

        # robot finishes first
        if robot_time < operator_time:
            
            # update state - robot
            self.robot_done[np.where(self.robot_task_id==self.robot_scheduled)[0][0]] = 1 # set the scheduled task as "done"
            self.robot_scheduled = 0
                
            if self.check_and_finish():
                return self.get_state(), initial_time-self.get_total_time(), True

        # operator finishes first
        elif robot_time > operator_time:

            # update state - operator
            self.operator_done[np.where(self.operator_task_id==self.operator_scheduled)[0][0]] = 1 # set the scheduled task as "done"
            self.operator_scheduled = 0
            
            if self.check_and_finish():
                return self.get_state(), initial_time-self.get_total_time(), True
            
        # robot and operator finish simultaneously
        else:

            # update state
            self.robot_done[np.where(self.robot_task_id==self.robot_scheduled)[0][0]] = 1 # set the scheduled task as "done"
            self.robot_scheduled = 0
            self.operator_done[np.where(self.operator_task_id==self.operator_scheduled)[0][0]] = 1 # set the scheduled task as "done"
            self.operator_scheduled = 0

            if self.check_and_finish():
                return self.get_state(), initial_time-self.get_total_time(), True

        # end of step
        return self.get_state(), initial_time-self.get_total_time(), False


    def check_and_finish(self) -> bool:
        """
        Check whether the episode can be concluded in this step.
        
        Returns
        -------
            bool
                True if the episode is over, False otherwise.
        """
        if self.robot_scheduled == 0 and self.operator_scheduled == 0:
            
            # Check whether robot and operator have no schedulable tasks in common, or one of them has no longer schedulable tasks
            
            robot_task_done = self.robot_task_id[self.robot_done == 1]
            operator_task_done = self.operator_task_id[self.operator_done == 1]
            task_scheduled_so_far = np.concatenate((robot_task_done, operator_task_done))
            
            robot_task_to_be_done = np.setdiff1d(self.robot_task_id, task_scheduled_so_far)
            operator_task_to_be_done = np.setdiff1d(self.operator_task_id, task_scheduled_so_far)

            # if robot and operator have no schedulable tasks in common (ie. one has no schedulable tasks left)
            if np.intersect1d(robot_task_to_be_done, operator_task_to_be_done).size == 0:
                
                self.robot_done[np.where(np.isin(self.robot_task_id, robot_task_to_be_done))[0]] = 1
                self.operator_done[np.where(np.isin(self.operator_task_id, operator_task_to_be_done))[0]] = 1
            
                return True # end of episode

            #if the there's a common task, but it's the last one left
            elif task_scheduled_so_far.size == 19:

                # if robot completes the last task faster than (or equal to) operator
                if self.robot_execution_time[np.where(self.robot_task_id==robot_task_to_be_done[0])[0][0]] <= \
                    self.operator_execution_time[np.where(self.operator_task_id==operator_task_to_be_done[0])[0][0]]:
                    
                    # robot completes the last task
                    self.robot_done[np.where(np.isin(self.robot_task_id, robot_task_to_be_done))[0]] = 1
                    
                else:
                    
                    # operator completes the last task    
                    self.operator_done[np.where(np.isin(self.operator_task_id, operator_task_to_be_done))[0]] = 1
                    
                # end of episode
                return True
                
            #if the there's a common task, but robot has only that task left, and it's waiting for it
            elif robot_task_to_be_done.size == 1:
               
                self.robot_done[np.where(np.isin(self.robot_task_id, robot_task_to_be_done))[0]] = 1
                self.operator_done[np.where(np.isin(self.operator_task_id, operator_task_to_be_done))[0]] = 1
                self.operator_done[np.where(np.isin(self.operator_task_id, robot_task_to_be_done))[0]] = 0

                # end of episode
                return True
                
        elif self.robot_scheduled == 0:

            # Check whether robot and operator have no schedulable tasks in common, or robot  has no longer a schedulable task
            
            robot_task_done = self.robot_task_id[self.robot_done == 1]
            operator_task_done = self.operator_task_id[self.operator_done == 1]
            task_scheduled_so_far = np.concatenate((robot_task_done, operator_task_done, [self.operator_scheduled]))
            
            robot_task_to_be_done = np.setdiff1d(self.robot_task_id, task_scheduled_so_far)
            operator_task_to_be_done = np.setdiff1d(self.operator_task_id, task_scheduled_so_far)

            # if robot and operator have no schedulable tasks in common
            if np.intersect1d(robot_task_to_be_done, operator_task_to_be_done).size == 0:

                # execute operator scheduled task
                self.operator_done[np.where(self.operator_task_id==self.operator_scheduled)[0][0]] = 1
                self.operator_scheduled = 0
            
                self.robot_done[np.where(np.isin(self.robot_task_id, robot_task_to_be_done))[0]] = 1
                self.operator_done[np.where(np.isin(self.operator_task_id, operator_task_to_be_done))[0]] = 1
            
                return True # end of episode

            #if the there's a common task, but robot has only that task left, and it's waiting for it
            elif robot_task_to_be_done.size == 1:
            
                # execute operator scheduled task
                self.operator_done[np.where(self.operator_task_id==self.operator_scheduled)[0][0]] = 1
                self.operator_scheduled = 0
               
                self.robot_done[np.where(np.isin(self.robot_task_id, robot_task_to_be_done))[0]] = 1           
                self.operator_done[np.where(np.isin(self.operator_task_id, operator_task_to_be_done))[0]] = 1
                self.operator_done[np.where(np.isin(self.operator_task_id, robot_task_to_be_done))[0]] = 0

                # end of episode
                return True

        else: #self.operator_scheduled == 0
            # Check whether robot and operator have no schedulable tasks in common
            
            robot_task_done = self.robot_task_id[self.robot_done == 1]
            operator_task_done = self.operator_task_id[self.operator_done == 1]
            task_scheduled_so_far = np.concatenate((robot_task_done, operator_task_done, [self.robot_scheduled]))
            
            robot_task_to_be_done = np.setdiff1d(self.robot_task_id, task_scheduled_so_far)
            operator_task_to_be_done = np.setdiff1d(self.operator_task_id, task_scheduled_so_far)

            # if robot and operator have no schedulable tasks in common
            if np.intersect1d(robot_task_to_be_done, operator_task_to_be_done).size == 0:

                # execute robot scheduled task
                self.robot_done[np.where(self.robot_task_id==self.robot_scheduled)[0][0]] = 1
                self.robot_scheduled = 0
            
                self.robot_done[np.where(np.isin(self.robot_task_id, robot_task_to_be_done))[0]] = 1
                self.operator_done[np.where(np.isin(self.operator_task_id, operator_task_to_be_done))[0]] = 1
            
                return True # end of episode

            #if the there's a common task, but operator has only that task left, and it's waiting for it
            elif operator_task_to_be_done.size == 1:
            
                # execute robot scheduled task
                self.robot_done[np.where(self.robot_task_id==self.robot_scheduled)[0][0]] = 1
                self.robot_scheduled = 0

                self.robot_done[np.where(np.isin(self.robot_task_id, robot_task_to_be_done))[0]] = 1
                self.robot_done[np.where(np.isin(self.robot_task_id, operator_task_to_be_done))[0]] = 0
                self.operator_done[np.where(np.isin(self.operator_task_id, operator_task_to_be_done))[0]] = 1
            
                # end of episode
                return True
        
        return False


    def get_valid_actions(self) -> torch.Tensor:
        """
        Generates a masking tensor for invalid actions based on the current state.

        Returns
        -------
            torch.Tensor
                1D tensor of size equal to the total number of possible actions,
                containing 1 for valid actions and 0 for invalid actions.
        """
        # initialize the masking tensor with all 0
        mask = torch.zeros(max(max(self.robot_task_id), max(self.operator_task_id))+1, dtype=torch.int64)
    
        # find actions no longer available
        robot_task_done = self.robot_task_id[self.robot_done == 1]
        operator_task_done = self.operator_task_id[self.operator_done == 1]
        tasks_done = np.concatenate((robot_task_done, operator_task_done))
    
        # if robot or both robot and operator have no scheduled tasks, find only actions schedulable for robot
        if self.robot_scheduled == 0:
            mask[self.robot_task_id] = 1  # robot schedulable tasks
        elif self.operator_scheduled == 0:
            mask[self.operator_task_id] = 1  # operator schedulable tasks
    
        # mask out already done actions
        mask[tasks_done] = 0
    
        # if a task is already scheduled, exclude it from the options
        if self.robot_scheduled != 0:
            mask[self.robot_scheduled] = 0
        if self.operator_scheduled != 0:
            mask[self.operator_scheduled] = 0
    
        #return mask
        return mask[1:] # return only 
    
            
    def sample_process_time(self) -> np.ndarray:
        """
        Samples processing time for each task for each operator.

        Returns
        -------
            np.ndarray
                Array of shape (n_operators x n_tasks) containing the processing time for each task for each operator.
        """
        return np.array([np.around(np.random.normal(loc=self.mu_operators[i], scale=self.std), decimals=3) for i in range(self.n_operators)])


    def set_operator(self, new_id_operator: int) -> None:
        """
        Set new operator and change operator execution time in state (to be called at the beginning of an episode).

        Parameters
        ----------
            id_operator: int
                Id of the new operator.
        """
        
        assert new_id_operator < self.n_operators, "Invalid Input!"

        # set new id operator
        self.id_operator = new_id_operator

        self.operator_execution_time = self.operators_sampled_time[self.id_operator]


    def is_over(self) -> bool:
        """
        Check whether the episode is over.

        Returns
        -------
            bool
                True whether the episode is over, False otherwise.
        """
        
        return np.sum(self.robot_done) + np.sum(self.operator_done) == 20.0 # sum of number tasks done by robot and tasks done by operators must be 20 at the end of an episode

    
    def get_total_time(self) -> float:
        """
        Return the total elapsed time of the episode.

        Returns
        -------
            float
                Episode elapsed time.
        """
        # robot total time
        r = np.sum(self.robot_done*self.robot_execution_time)
        # operator total time
        t = np.sum(self.operator_done*self.operator_execution_time)
        
        return max(r,t)


    def get_state(self) -> tuple:
        """
        Return the current state of the environment.

        Returns
        -------
            tuple
                Current state.
        """
        return self.robot_done, self.robot_scheduled, self.robot_execution_time, self.operator_done, self.operator_scheduled, self.operator_execution_time