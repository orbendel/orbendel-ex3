//
// Created by orbendel on 06/06/2021.
//
#include "MapReduceFramework.h"
#include <vector>
#include <pthread.h>
#include <algorithm>
#include <atomic>
#include <semaphore.h>
#include "Barrier.h"
#include <iostream>

using namespace std;

typedef struct JobContext jobcontext;

typedef struct {
    int threadID;
    JobContext *jobContext;
    IntermediateVec *intermediateVec;
    pthread_t thread;
}ThreadContext;

struct JobContext{
    const MapReduceClient *client;
    const InputVec inputVec;
    unsigned long shuffleCounter;
    OutputVec *outputVec;
    int multiThreadLevel;
    vector<ThreadContext> threadsContext;
    atomic<uint64_t> atomicVariable;
    Barrier barrier;
    vector<vector<IntermediatePair>> shuffleVec;
    sem_t semForShuffle;
    pthread_mutex_t mutex_reduce;
    bool jobDone;
    unsigned long pairsCount;
    unsigned long toReduce;

    JobContext(const MapReduceClient& client,int multiThreadLevel, const InputVec& inputVec,
               OutputVec& outputVec, void * f(void *)): client(&client),inputVec(inputVec), outputVec(&outputVec) ,
               multiThreadLevel(multiThreadLevel),barrier(multiThreadLevel),
               atomicVariable(0), jobDone(false),shuffleCounter(-1),toReduce(-1)

    {
        int sem_ret = sem_init(&this->semForShuffle, 0, 0);
        if(sem_ret != 0)
        {
            cerr << "system error: sem fail\n";
            delete this;
            exit(1);
        }
        pairsCount = inputVec.size();
        mutex_reduce = PTHREAD_MUTEX_INITIALIZER;
        try
        {
            for(int i = 0; i< multiThreadLevel; i++)
            {
                ThreadContext *tx = new ThreadContext();
                tx->threadID = i;
                tx->intermediateVec = new IntermediateVec();
                tx->jobContext = this;
                threadsContext.push_back(*tx);
            }

            for(int i = 0; i<multiThreadLevel; i++)
            {
                if(pthread_create(&threadsContext[i].thread, nullptr, f, &threadsContext[i]) != 0)
                {
                    cerr << "system error: thread creation fail\n";
                    delete this;
                    exit(1);
                }
            }

        }
        catch (...) {
            cerr << "system error: thread creation fail\n";
            delete this;
            exit(1);
        }
    }

    ~JobContext()
    {
        for(auto t:threadsContext)
        {
            //pthread_cancel(t.thread);
            delete t.intermediateVec;
        }
        threadsContext.clear();
        pthread_mutex_destroy(&mutex_reduce);
        sem_destroy(&semForShuffle);
    }
};


void mapPhase(ThreadContext *threadContext)
{
    JobContext *curJobContext = threadContext->jobContext;
    if(curJobContext->atomicVariable.load() >> 62 ==0){
        (curJobContext->atomicVariable) += uint64_t(1) << 62;
    }
    int old_value_idx = curJobContext->atomicVariable++;
    old_value_idx = old_value_idx & (0x7fffffff);
    while(old_value_idx < curJobContext->inputVec.size())
    {
        InputPair curPair = curJobContext->inputVec[old_value_idx];
        curJobContext->client->map(curPair.first, curPair.second, threadContext);
        old_value_idx = curJobContext->atomicVariable++;
        old_value_idx = old_value_idx & (0x7fffffff);
    }
}

/**
 * Comparing the first element of the pair
 * @param x IntermediatePair
 * @param y IntermediatePair
 * @return by the operator from the user
 */
bool compareIntermediate(IntermediatePair x, IntermediatePair y)
{
    return *(x.first) < *(y.first);
}
/**
 *
 * @param threadContext
 */
void sortPhase(ThreadContext *threadContext)
{
    sort(threadContext->intermediateVec->begin(), threadContext->intermediateVec->end(),&compareIntermediate);
    threadContext->jobContext->barrier.barrier();
}

/**
 * @return true if there are any vectors of threads left to shuffle, false otherwise
 */
bool vectorsToShuffle(vector<ThreadContext> threadsContext)
{
    for(int i = 0; i < threadsContext.size(); i++)
        if(threadsContext[i].intermediateVec->empty() == false)
            return true;
    return true;
}

/**
 * @param threadsContext threads contexts vector
 * @return the largest key of all the threads contexts after mapping
 */
K2* getLargestKey(vector<ThreadContext> threadsContext,ThreadContext *tx )
{
    unsigned long counter = 0;
    K2* largestKey = nullptr;
    for(int i = 0; i < threadsContext.size(); i++)
    {
        counter += threadsContext[i].intermediateVec->size();
        if(threadsContext[i].intermediateVec->empty())
            continue;

        else if(largestKey == nullptr || *largestKey < *threadsContext[i].intermediateVec->back().first){
            largestKey = threadsContext[i].intermediateVec->back().first;
        }
    }
    if(tx->jobContext->shuffleCounter == -1)
    {
        tx->jobContext->shuffleCounter = counter;
    }
    return largestKey;
}
/**
 *
 * @param threadContext
 */
void shufflePhase(ThreadContext *threadContext)
{
    threadContext->jobContext->atomicVariable =uint64_t(2) << 62;
    JobContext *jobContext = threadContext->jobContext;
    vector<ThreadContext> threadsContext = jobContext->threadsContext;

    K2* largestKey;
    IntermediateVec curVec;

    while(vectorsToShuffle(threadsContext))
    {
        largestKey = getLargestKey(threadsContext,threadContext);
        curVec.clear();
        for(int i=0; i<threadsContext.size(); i++ )
        {
            while(threadsContext[i].intermediateVec->empty() == false &&
                  !((threadsContext[i].intermediateVec->back().first) < largestKey) &&
                  !((threadsContext[i].intermediateVec->back().first) > largestKey) )
            {
                curVec.push_back(threadsContext[i].intermediateVec->back());
                threadContext->jobContext->atomicVariable++ ;
                threadsContext[i].intermediateVec->pop_back();

            }
        }

        jobContext->shuffleVec.push_back(curVec);
    }
}

void reducePhase(ThreadContext *threadContext)
{
    threadContext->jobContext->toReduce = threadContext->jobContext->shuffleVec.size();
    while(!threadContext->jobContext->shuffleVec.empty()) {
        if(pthread_mutex_lock(&threadContext->jobContext->mutex_reduce) != 0)
        {
            cerr << "system error: mutex fail\n";
            delete threadContext->jobContext;
            exit(1);
        }
        threadContext->jobContext->client->reduce(&threadContext->jobContext->shuffleVec.back(), threadContext);
        threadContext->jobContext->atomicVariable += uint64_t (1) << 31;
        threadContext->jobContext->shuffleVec.pop_back();
        if(pthread_mutex_unlock(&threadContext->jobContext->mutex_reduce) != 0)
        {
            cerr << "system error: mutex fail\n";
            delete threadContext->jobContext;
            exit(1);
        }
    }
}
/**
 *
 * @param threadContext
 */
void* flowFunction(void* context)
{
    auto* threadContext = (ThreadContext *)context;
    mapPhase(threadContext);
    sortPhase(threadContext);

    if(threadContext->threadID == 0)
    {
        shufflePhase(threadContext);
        if(sem_post(&threadContext->jobContext->semForShuffle) != 0)
        {
            cerr << "system error: sem fail\n";
            delete threadContext->jobContext;
            exit(1);
        }
    }
   // if(sem_wait(&threadContext->jobContext->semForShuffle) != 0)
    //{
    //    cerr << "system error: sem fail\n";
    //    delete threadContext->jobContext;
    //    exit(1);
   // }
    threadContext->jobContext->atomicVariable = uint64_t(3) << 62;
    threadContext->jobContext->barrier.barrier();
    reducePhase(threadContext);
    threadContext->jobContext->jobDone = true;
    return nullptr;
}

/**
 *
 * @param key
 * @param value
 * @param context
 */
void emit2 (K2* key, V2* value, void* context)
{
    ThreadContext* input_context = (ThreadContext *)context;
    input_context->intermediateVec->push_back({key,value});
}

/**
 *
 * @param key
 * @param value
 * @param context
 */
void emit3 (K3* key, V3* value, void* context)
{
    auto * threadContext = (ThreadContext *)context;
    if(pthread_mutex_lock(&threadContext->jobContext->mutex_reduce) != 0)
    {
        cerr << "system error: mutex fail\n";
        delete threadContext->jobContext;
        exit(1);
    }
    threadContext->jobContext->outputVec->push_back({key,value});
    if(pthread_mutex_unlock(&threadContext->jobContext->mutex_reduce) != 0)
    {
        cerr << "system error: mutex fail\n";
        delete threadContext->jobContext;
        exit(1);
    }

}

JobHandle startMapReduceJob(const MapReduceClient& client,
                            const InputVec& inputVec, OutputVec& outputVec,
                            int multiThreadLevel)
{
    auto jobContext = new JobContext(client,multiThreadLevel,inputVec,outputVec,(void *(*)(void *))(&flowFunction));
    return static_cast<JobHandle>(jobContext);
}

/**
 *
 * @param job
 */
void waitForJob(JobHandle job)
{
    JobContext* jobInfo = (JobContext*) job;
    if(!jobInfo->jobDone)
    {
        for(int i = 0; i < jobInfo->multiThreadLevel; i++)
        {
            if(pthread_join(jobInfo->threadsContext[i].thread, nullptr) != 0)
            {
                cerr << "system error: thread join fail\n";
                delete jobInfo;
                exit(1);
            }
        }
    }
}

/**
 *
 * @param job
 * @param state
 */
void getJobState(JobHandle job, JobState* state)
{
    JobContext* context = (JobContext *)job;
    unsigned  long atomicVar = context->atomicVariable.load();
    auto stateIn = atomicVar >> 62;
    if(stateIn == 0)
    {
        state->stage = UNDEFINED_STAGE;
        state->percentage = 0;
    }else if(stateIn == 1)
    {
        state->stage = MAP_STAGE;
        state->percentage = 100 * ((float) ((atomicVar >> 31) & (0x7fffffff)))/ (float)context->inputVec.size();

    }else if(stateIn == 2)
    {
        state->stage = SHUFFLE_STAGE;
        if (context->pairsCount == 0){
            state->percentage = 0;
        } else
        {
            state -> percentage = 100 * (float)(atomicVar & (0x7fffffff)) / (float)context->shuffleCounter;
        }
    }else // == 3
    {
        state->stage = REDUCE_STAGE;
        if(context->shuffleVec.size() == 0)
        {
            state->percentage = 0;
        }else
        {
            if(context->toReduce == 0)
            {
                state->percentage = 0;
            }
            else{
                state->percentage = 100 * ((float) ((atomicVar >> 31)& (0x7fffffff))) / (float) context->toReduce;
            }

        }
    }
}

/**
 *
 * @param job
 */
void closeJobHandle(JobHandle job)
{
    JobContext* jobContext = (JobContext *)job;
    if(!jobContext->jobDone)
        waitForJob(job);
    delete static_cast<JobContext *>(jobContext);
}