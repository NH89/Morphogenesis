
#include <vtkCellArray.h>
#include <vtkNew.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkXMLPolyDataWriter.h>

int main(int, char*[])
{
  // Create 10 points.
  vtkNew<vtkPoints> points;

  for (unsigned int i = 0; i < 10; ++i)
  {
    points->InsertNextPoint(i, i, i);
  }

  // Create a polydata object and add the points to it.
  vtkNew<vtkPolyData> polydata;
  polydata->SetPoints(points);

  // Write the file
  vtkNew<vtkXMLPolyDataWriter> writer;
  writer->SetFileName("test.vtp");
  writer->SetInputData(polydata);

  // Optional - set the mode. The default is binary.
  // writer->SetDataModeToBinary();
  // writer->SetDataModeToAscii();

  writer->Write();


  char relativePath[6] = "check";
  int frame = 1;
																									cout<<"\n"<<*polydata<<std::flush;
																									cout<<"\nFluidSystem::SavePointsVTP2  chk 19"<<std::flush;
    char buf[256];
    frame += 100000;                                                                                // ensures numerical and alphabetic order match of filenames
    sprintf ( buf, "%s/particles_pos_vel_color%04d.vtp", relativePath, frame );
	writer->SetFileName(buf);
    																								cout<<"\n"<<writer->GetFileName()<<std::flush;
																									cout<<"\nFluidSystem::SavePointsVTP2  chk 20"<<std::flush;
	writer->SetInputData(polydata);
    																								cout<<"\n"<<*writer<<std::flush;
																									cout<<"\nFluidSystem::SavePointsVTP2  chk 21\n"<<std::flush;
    //writer->SetDataModeToAscii();
    writer->SetDataModeToAppended();    // prefered, produces a human readable header followed by a binary blob.
																									cout<<"\nFluidSystem::SavePointsVTP2  chk 22\n"<<std::flush;
    //writer->SetDataModeToBinary();
	writer->Write();
																									cout<<"\nFluidSystem::SavePointsVTP2  chk 23"<<std::flush;


  return EXIT_SUCCESS;
}
