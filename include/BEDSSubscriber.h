// BEGINLICENSE
//
// This file is part of apoCHARMM, which is distributed under the BSD 3-clause
// license, as described in the LICENSE file in the top level directory of this
// project.
//
// Author:  Samarjeet Prasad
//
// ENDLICENSE

#pragma once

#include "Subscriber.h"
#include <fstream>

class BEDSSubscriber : public Subscriber {
public:
  BEDSSubscriber(const std::string &fileName);
  BEDSSubscriber(const std::string &fileName, const int reportFrequency);
  ~BEDSSubscriber(void);

public:
  void update(void) override;

private:
  int m_NumFramesWritten;
};
