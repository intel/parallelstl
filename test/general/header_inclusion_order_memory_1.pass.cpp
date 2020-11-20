// -*- C++ -*-
//===-- header_inclusion_order_memory_1.pass.cpp --------------------------===//
//
// Copyright (C) 2017-2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#include "support/pstl_test_config.h"

#include _PSTL_TEST_HEADER(memory)
#include _PSTL_TEST_HEADER(execution)

#include "support/utils.h"

int
main()
{
    ::std::cout << TestUtils::done() << ::std::endl;
    return 0;
}
