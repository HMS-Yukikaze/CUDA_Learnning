#include "manager.hpp"


Manager::Manager(std::string _file, size_t _sz)
	:engine{ _file },
	detector{ std::make_shared<Detect>() },
	buf{ _sz }
{
}

Manager::~Manager()
{
}