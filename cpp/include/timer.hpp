#ifndef TIMER_HPP
#define TIMER_HPP

#include <time.h>

class Timer
{
private:
    double start_time;

    static double get_time()
    {
        timespec t;
        clock_gettime(CLOCK_REALTIME, &t);
        return t.tv_sec + double(t.tv_nsec) * 1e-9;
    }

public:
    explicit Timer() : start_time(get_time()) {}

    double get() const
    {
        return get_time() - start_time;
    }
};

#endif
