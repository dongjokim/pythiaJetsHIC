#ifndef FASTJET_COMPAT_H
#define FASTJET_COMPAT_H

#include <memory>

// Create a compatibility layer for std::auto_ptr
namespace std {
    template<typename T>
    using auto_ptr = unique_ptr<T>;
}

#endif // FASTJET_COMPAT_H 